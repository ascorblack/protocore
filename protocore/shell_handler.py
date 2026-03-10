"""Shell-specific orchestration helpers extracted from `orchestrator.py`."""

from __future__ import annotations

import json
import logging
import re
import shlex
from typing import Any, Callable

from .constants import BLOCKED_SHELL_ENV_KEYS, MAX_SHELL_ENV_VALUE_LENGTH
from .context import validate_path_access
from .orchestrator_errors import ContractViolationError, PendingShellApprovalError
from .orchestrator_utils import PolicyRunner
from .protocols import ShellExecutor, ShellSafetyPolicy
from .shell_safety import build_shell_execution_hash, build_shell_payload_hash
from .types import (
    AgentContext,
    AgentContextMeta,
    ExecutionReport,
    ExecutionStatus,
    Message,
    PolicyDecision,
    Result,
    ShellApprovalRule,
    ShellCommandPlan,
    ShellExecutionRequest,
    ShellToolConfig,
    StopReason,
    ToolContext,
    ToolResult,
    ToolResultMeta,
)

logger = logging.getLogger(__name__)


def resume_from_pending(
    pending: Result | dict[str, Any],
    decision: str | bool,
) -> dict[str, dict[str, Any]]:
    """Build the metadata patch required to resume a pending shell approval."""
    metadata = pending.metadata if isinstance(pending, Result) else pending
    pending_plan = metadata.get(AgentContextMeta.PENDING_SHELL_APPROVAL)
    if not isinstance(pending_plan, dict):
        raise ValueError(
            "resume_from_pending requires Result.metadata['pending_shell_approval']"
        )
    plan_id = pending_plan.get("plan_id")
    if not isinstance(plan_id, str) or not plan_id:
        raise ValueError("pending_shell_approval must include non-empty plan_id")
    return {
        AgentContextMeta.PENDING_SHELL_APPROVAL: pending_plan,
        AgentContextMeta.SHELL_APPROVAL_DECISIONS: {plan_id: decision},
    }


class ShellHandler:
    """Owns shell request normalization and approval resumption."""

    def __init__(
        self,
        *,
        shell_executor: ShellExecutor | None,
        policy_runner: PolicyRunner,
        shell_safety_policy: ShellSafetyPolicy | None = None,
        append_tool_results_as_messages: Callable[[list[Message], list[ToolResult]], None],
    ) -> None:
        self._shell_executor = shell_executor
        self._policy_runner = policy_runner
        self._shell_safety_policy = shell_safety_policy
        self._append_tool_results_as_messages = append_tool_results_as_messages

    @staticmethod
    def normalize_shell_request(
        arguments: dict[str, Any],
        capability: ShellToolConfig,
        tool_context: ToolContext,
    ) -> ShellExecutionRequest:
        request = ShellExecutionRequest.model_validate(arguments)
        timeout_ms = request.timeout_ms or capability.default_timeout_ms
        if timeout_ms > capability.max_timeout_ms:
            raise ValueError("shell timeout exceeds shell_tool_config.max_timeout_ms")
        if len(request.command) > capability.max_command_length:
            raise ValueError("shell command exceeds shell_tool_config.max_command_length")
        if capability.require_cwd and not request.cwd:
            raise ValueError("shell cwd is required by shell_tool_config")
        normalized_cwd = request.cwd
        if request.cwd is not None:
            normalized_cwd = str(validate_path_access(request.cwd, tool_context))
        if request.env:
            allowed_env = set(capability.env_allowlist)
            disallowed = sorted(key for key in request.env if key not in allowed_env)
            if disallowed:
                raise ValueError(
                    "shell env keys are not allowlisted: " + ",".join(disallowed)
                )
            reserved = sorted(
                key for key in request.env if key in BLOCKED_SHELL_ENV_KEYS
            )
            if reserved:
                raise ValueError(
                    "shell env keys are reserved and cannot be set: "
                    + ",".join(reserved)
                )
            too_long = sorted(
                key
                for key, value in request.env.items()
                if len(value) > MAX_SHELL_ENV_VALUE_LENGTH
            )
            if too_long:
                raise ValueError(
                    "shell env values exceed max length for: " + ",".join(too_long)
                )
        return request.model_copy(update={"timeout_ms": timeout_ms, "cwd": normalized_cwd})

    @staticmethod
    def shell_result_to_tool_result(
        *,
        tool_call_id: str,
        tool_name: str,
        request: ShellExecutionRequest,
        result: Any,
        capability: ShellToolConfig,
    ) -> ToolResult:
        payload: dict[str, Any] = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "truncated": result.truncated,
        }
        if result.duration_ms is not None:
            payload["duration_ms"] = result.duration_ms
        metadata = dict(result.metadata)
        metadata.update(
            {
                "shell_command": request.command,
                "shell_cwd": request.cwd,
                "shell_profile": capability.profile.value,
                "shell_network_allowed": capability.allow_network,
                ToolResultMeta.SHELL_RISK_FLAGS: list(result.risk_flags),
                "shell_exit_code": result.exit_code,
            }
        )
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=json.dumps(payload, ensure_ascii=True),
            is_error=result.exit_code != 0,
            metadata=metadata,
        )

    @staticmethod
    def _compute_payload_hash(request: ShellExecutionRequest) -> str:
        """Compute a deterministic hash of the shell request payload."""
        return build_shell_payload_hash(request)

    @staticmethod
    def _compute_execution_hash(request: ShellExecutionRequest) -> str:
        """Compute a guard hash that includes resolved executable identity."""
        return build_shell_execution_hash(request)

    @staticmethod
    def _parse_approval_decision(
        decision_raw: Any,
    ) -> tuple[bool, bool, bool, ShellApprovalRule | None]:
        approved = False
        rejected = False
        add_to_session_allowlist = False
        approval_rule: ShellApprovalRule | None = None
        if isinstance(decision_raw, bool):
            return decision_raw is True, decision_raw is False, False, None
        if isinstance(decision_raw, str):
            normalized_decision = decision_raw.strip().lower()
            return (
                normalized_decision in {"approve", "approved", "allow"},
                normalized_decision in {"reject", "rejected", "deny", "denied"},
                False,
                None,
            )
        if not isinstance(decision_raw, dict):
            return approved, rejected, add_to_session_allowlist, approval_rule
        decision_value = decision_raw.get("decision")
        if isinstance(decision_value, bool):
            approved = decision_value is True
            rejected = decision_value is False
        elif isinstance(decision_value, str):
            normalized_decision = decision_value.strip().lower()
            approved = normalized_decision in {"approve", "approved", "allow"}
            rejected = normalized_decision in {"reject", "rejected", "deny", "denied"}
        add_to_session_allowlist = bool(
            decision_raw.get("add_to_session_allowlist", False)
        )
        approval_rule_raw = decision_raw.get("approval_rule")
        if isinstance(approval_rule_raw, dict):
            try:
                approval_rule = ShellApprovalRule.model_validate(approval_rule_raw)
            except Exception:
                logger.warning("Ignoring invalid shell approval rule payload")
        return approved, rejected, add_to_session_allowlist, approval_rule

    @staticmethod
    def build_similar_approval_rule(
        pending_plan: ShellCommandPlan,
    ) -> ShellApprovalRule:
        """Build a session rule from a previously approved shell command."""
        command_pattern = rf"^{re.escape(pending_plan.command)}(?:\s+.*)?$"
        pattern_kind = "exact_command_prefix"
        try:
            tokens = shlex.split(pending_plan.command, posix=True)
        except ValueError:
            tokens = []
        if tokens:
            executable = tokens[0].strip()
            if executable:
                command_pattern = rf"^\s*{re.escape(executable)}(?:\s|$)"
                pattern_kind = "same_executable"
        cwd_pattern = None
        if isinstance(pending_plan.cwd, str) and pending_plan.cwd:
            cwd_pattern = rf"^{re.escape(pending_plan.cwd)}$"
        return ShellApprovalRule(
            tool_name_pattern=rf"^{re.escape(pending_plan.tool_name)}$",
            command_pattern=command_pattern,
            cwd_pattern=cwd_pattern,
            description=(
                f"Allow shell commands matching {command_pattern}"
                + (f" in cwd={pending_plan.cwd}" if pending_plan.cwd else "")
            ),
            added_via="session_approval",
            metadata={
                "derived_from_plan_id": pending_plan.plan_id,
                "pattern_kind": pattern_kind,
            },
        )

    @staticmethod
    def _matches_rule(
        pending_plan: ShellCommandPlan,
        rule: ShellApprovalRule,
    ) -> bool:
        comparisons: tuple[tuple[str | None, str | None], ...] = (
            (rule.tool_name_pattern, pending_plan.tool_name),
            (rule.command_pattern, pending_plan.command),
            (rule.cwd_pattern, pending_plan.cwd),
            (rule.reason_pattern, pending_plan.reason),
        )
        for pattern, value in comparisons:
            if pattern is None:
                continue
            if value is None or re.search(pattern, value) is None:
                return False
        return True

    @classmethod
    def find_matching_approval_rule(
        cls,
        context: AgentContext,
        pending_plan: ShellCommandPlan,
    ) -> ShellApprovalRule | None:
        rules_raw = context.metadata.get(AgentContextMeta.SHELL_APPROVAL_RULES, [])
        if not isinstance(rules_raw, list):
            return None
        for item in rules_raw:
            if not isinstance(item, dict):
                continue
            try:
                rule = ShellApprovalRule.model_validate(item)
            except Exception:
                logger.warning("Ignoring invalid shell approval rule in context metadata")
                continue
            if cls._matches_rule(pending_plan, rule):
                return rule
        return None

    @staticmethod
    def add_session_approval_rule(
        context: AgentContext,
        rule: ShellApprovalRule,
    ) -> None:
        rules_raw = context.metadata.setdefault(AgentContextMeta.SHELL_APPROVAL_RULES, [])
        if not isinstance(rules_raw, list):
            rules_raw = []
            context.metadata[AgentContextMeta.SHELL_APPROVAL_RULES] = rules_raw
        rule_dump = rule.model_dump(mode="json")
        if rule_dump not in rules_raw:
            rules_raw.append(rule_dump)

    @staticmethod
    def build_shell_command_plan(
        *,
        tool_call_id: str,
        tool_name: str,
        request: ShellExecutionRequest,
        capability: ShellToolConfig,
    ) -> ShellCommandPlan:
        payload_hash = ShellHandler._compute_payload_hash(request)
        execution_hash = ShellHandler._compute_execution_hash(request)
        return ShellCommandPlan(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            command=request.command,
            cwd=request.cwd,
            timeout_ms=request.timeout_ms,
            env=request.env,
            reason=request.reason,
            metadata={
                "shell_profile": capability.profile.value,
                "shell_network_allowed": capability.allow_network,
                "payload_hash": payload_hash,
                "execution_hash": execution_hash,
            },
        )

    async def resume_pending_shell_approval(
        self,
        *,
        messages: list[Message],
        context: AgentContext,
        report: ExecutionReport,
        pending_plan: ShellCommandPlan,
    ) -> None:
        decision_map = context.metadata.get(AgentContextMeta.SHELL_APPROVAL_DECISIONS, {})
        decision_raw: Any = None
        if isinstance(decision_map, dict):
            decision_raw = decision_map.get(pending_plan.plan_id)

        approved, rejected, add_to_session_allowlist, approval_rule = (
            self._parse_approval_decision(decision_raw)
        )

        if approved:
            pending_plan.transition_to("approved")
            capability = context.config.shell_tool_config
            request = self.normalize_shell_request(
                {
                    "command": pending_plan.command,
                    "cwd": pending_plan.cwd,
                    "timeout_ms": pending_plan.timeout_ms,
                    "env": pending_plan.env,
                    "reason": pending_plan.reason,
                },
                capability,
                context.tool_context,
            )
            # Verify payload integrity: the approved plan must match
            original_hash = (pending_plan.metadata or {}).get("payload_hash")
            if original_hash is not None:
                current_hash = self._compute_payload_hash(request)
                if current_hash != original_hash:
                    tool_result = ToolResult(
                        tool_call_id=pending_plan.tool_call_id,
                        tool_name=pending_plan.tool_name,
                        content="[SHELL APPROVAL REJECTED: payload was modified after approval]",
                        is_error=True,
                        metadata={"shell_approval_payload_mismatch": True},
                    )
                    report.shell_approvals_rejected += 1
                    pending_plan.transition_to("rejected")
                    self._append_tool_results_as_messages(messages, [tool_result])
                    context.metadata.pop(AgentContextMeta.PENDING_SHELL_APPROVAL, None)
                    if isinstance(decision_map, dict):
                        decision_map.pop(pending_plan.plan_id, None)
                    return
            original_execution_hash = (pending_plan.metadata or {}).get("execution_hash")
            if original_execution_hash is not None:
                current_execution_hash = self._compute_execution_hash(request)
                if current_execution_hash != original_execution_hash:
                    tool_result = ToolResult(
                        tool_call_id=pending_plan.tool_call_id,
                        tool_name=pending_plan.tool_name,
                        content="[SHELL APPROVAL REJECTED: execution context changed after approval]",
                        is_error=True,
                        metadata={"shell_approval_execution_hash_mismatch": True},
                    )
                    report.shell_approvals_rejected += 1
                    pending_plan.transition_to("rejected")
                    self._append_tool_results_as_messages(messages, [tool_result])
                    context.metadata.pop(AgentContextMeta.PENDING_SHELL_APPROVAL, None)
                    if isinstance(decision_map, dict):
                        decision_map.pop(pending_plan.plan_id, None)
                    return
            # Re-run shell safety policy to catch config/policy changes
            if (
                capability.safety_mode.value != "yolo"
                and self._shell_safety_policy is not None
            ):
                re_decision = await self._shell_safety_policy.evaluate(
                    request, context.tool_context, capability,
                )
                if re_decision == PolicyDecision.DENY:
                    logger.warning(
                        "Shell resume rejected by safety policy re-evaluation: plan_id=%s",
                        pending_plan.plan_id,
                    )
                    tool_result = ToolResult(
                        tool_call_id=pending_plan.tool_call_id,
                        tool_name=pending_plan.tool_name,
                        content="[SHELL APPROVAL REJECTED: safety policy now denies this command]",
                        is_error=True,
                        metadata={"shell_approval_policy_changed": True},
                    )
                    report.shell_approvals_rejected += 1
                    pending_plan.transition_to("rejected")
                    self._append_tool_results_as_messages(messages, [tool_result])
                    context.metadata.pop(AgentContextMeta.PENDING_SHELL_APPROVAL, None)
                    if isinstance(decision_map, dict):
                        decision_map.pop(pending_plan.plan_id, None)
                    return
            if self._shell_executor is None:
                raise ContractViolationError(
                    "SHELL_EXECUTOR_REQUIRED",
                    "Pending shell approval cannot be resumed without ShellExecutor",
                )
            if add_to_session_allowlist or approval_rule is not None:
                self.add_session_approval_rule(
                    context,
                    approval_rule or self.build_similar_approval_rule(pending_plan),
                )
            _executor = self._shell_executor
            async def _execute_shell(
                _req: Any = request,
                _ctx: Any = context.tool_context,
                _cap: Any = capability,
            ) -> Any:
                return await _executor.execute(_req, _ctx, _cap)

            shell_result = await self._policy_runner.call(
                operation="tool.shell_execute",
                report=report,
                retryable=False,
                fn=_execute_shell,
            )
            tool_result = self.shell_result_to_tool_result(
                tool_call_id=pending_plan.tool_call_id,
                tool_name=pending_plan.tool_name,
                request=request,
                result=shell_result,
                capability=capability,
            )
            for flag in tool_result.metadata.get(ToolResultMeta.SHELL_RISK_FLAGS, []):
                if isinstance(flag, str) and flag not in report.shell_risk_flags:
                    report.shell_risk_flags.append(flag)
            pending_plan.transition_to("executed")
            report.shell_approvals_granted += 1
            self._append_tool_results_as_messages(messages, [tool_result])
            context.metadata.pop(AgentContextMeta.PENDING_SHELL_APPROVAL, None)
            if isinstance(decision_map, dict):
                decision_map.pop(pending_plan.plan_id, None)
            return

        if rejected:
            pending_plan.transition_to("rejected")
            tool_result = ToolResult(
                tool_call_id=pending_plan.tool_call_id,
                tool_name=pending_plan.tool_name,
                content="[SHELL APPROVAL REJECTED BY USER]",
                is_error=True,
                metadata={"shell_approval_rejected": True},
            )
            report.shell_approvals_rejected += 1
            self._append_tool_results_as_messages(messages, [tool_result])
            context.metadata.pop(AgentContextMeta.PENDING_SHELL_APPROVAL, None)
            if isinstance(decision_map, dict):
                decision_map.pop(pending_plan.plan_id, None)
            return

        report.finalize(
            ExecutionStatus.PARTIAL,
            stop_reason=StopReason.APPROVAL_REQUIRED,
            error_code="APPROVAL_REQUIRED",
            error_message=f"Shell approval pending for plan_id={pending_plan.plan_id}",
        )
        raise PendingShellApprovalError(pending_plan)


__all__ = ["ShellHandler"]
