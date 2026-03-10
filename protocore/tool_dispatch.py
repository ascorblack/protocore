"""Tool dispatch helpers extracted from `orchestrator.py`."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .constants import LOAD_SKILL_TOOL_NAME
from .context import PathIsolationError, contains_path_argument, validate_path_arguments
from .events import (
    EV_DESTRUCTIVE_ACTION,
    EV_INJECTION_SIGNAL,
    EV_SKILL_BUDGET_EXCEEDED,
    EV_SKILL_LOAD_END,
    EV_SKILL_LOAD_START,
    EV_TOOL_APPROVAL_REQUIRED,
    EV_TOOL_APPROVAL_RESOLVED,
    EV_TOOL_CALL_END,
    EV_TOOL_CALL_FAILED,
    EV_TOOL_CALL_START,
    EV_TOOL_DISPATCH_SELECTED,
    EV_TOOL_EXECUTION_END,
    EV_TOOL_EXECUTION_START,
    EV_TOOL_PREFLIGHT_END,
    EV_TOOL_PREFLIGHT_START,
    EV_TOOL_RESULT_READY,
    EV_TOOL_VALIDATION_END,
    EV_TOOL_VALIDATION_START,
    EventBus,
)
from .hooks.manager import HookManager
from .orchestrator_errors import PendingShellApprovalError
from .orchestrator_utils import (
    PolicyRunner,
    event_context_payload,
    normalize_policy_decision,
    redact_sensitive_keys,
    resolve_shell_capability,
    string_preview,
    tool_payload_summary,
)
from .protocols import (
    ExecutionPolicy,
    ShellExecutor,
    ShellSafetyPolicy,
    ToolExecutor,
)
from .registry import ToolRegistry
from .shell_handler import ShellHandler
from .types import (
    AgentContext,
    ExecutionReport,
    PolicyDecision,
    RunKind,
    ShellToolConfig,
    ToolCall,
    ToolContextMeta,
    ToolResult,
    ToolResultMeta,
)

logger = logging.getLogger(__name__)


class ToolDispatcher:
    """Encapsulates tool preflight and execution dispatch."""

    def __init__(
        self,
        *,
        hooks: HookManager,
        event_bus: EventBus,
        policy: ExecutionPolicy | None,
        tool_registry: ToolRegistry | None,
        tool_executor: ToolExecutor | None,
        shell_executor: ShellExecutor | None,
        shell_safety_policy: ShellSafetyPolicy | None,
        shell_handler: ShellHandler,
        policy_runner: PolicyRunner,
    ) -> None:
        self._hooks = hooks
        self._bus = event_bus
        self._policy = policy
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._shell_executor = shell_executor
        self._shell_safety_policy = shell_safety_policy
        self._shell_handler = shell_handler
        self._policy_runner = policy_runner

    @staticmethod
    def _tool_invocation_payload(
        *,
        arguments: dict[str, Any],
        is_shell_tool: bool,
    ) -> dict[str, Any]:
        redacted_arguments = redact_sensitive_keys(arguments)
        if not isinstance(redacted_arguments, dict):
            redacted_arguments = {}
        payload: dict[str, Any] = {
            "arguments": redacted_arguments,
            "argument_keys": sorted(arguments.keys()),
            "arguments_json": json.dumps(
                redacted_arguments,
                ensure_ascii=True,
                sort_keys=True,
                default=str,
            ),
        }
        if is_shell_tool:
            shell_command = arguments.get("command")
            if isinstance(shell_command, str):
                payload["shell_command"] = shell_command
            shell_cwd = arguments.get("cwd")
            if isinstance(shell_cwd, str):
                payload["shell_cwd"] = shell_cwd
        return payload

    async def dispatch_tool(
        self,
        tool_call: Any,
        context: AgentContext,
        report: ExecutionReport,
        run_kind: RunKind,
    ) -> ToolResult:
        normalized = self._normalize_tool_call(tool_call=tool_call, report=report)
        if isinstance(normalized, ToolResult):
            return normalized
        tool_call_id, tool_name, raw_arguments = normalized
        try:
            arguments = self._parse_tool_arguments(
                tool_name=tool_name,
                raw_args=raw_arguments,
                report=report,
            )
        except ValueError as exc:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                content=f"[MALFORMED TOOL CALL: {exc}]",
                is_error=True,
                metadata={"tool_call_malformed": True},
            )
        shell_tool_name = context.config.shell_tool_config.tool_name
        shell_capability = resolve_shell_capability(context.config, run_kind)
        is_shell_tool = (
            shell_capability is not None and tool_name == shell_capability.tool_name
        )
        registry_definition = (
            self._tool_registry.get_definition(tool_name)
            if self._tool_registry
            else None
        )
        config_definition = next(
            (
                definition
                for definition in context.config.tool_definitions
                if definition.name == tool_name
            ),
            None,
        )

        matches_shell_tool_name = tool_name == shell_tool_name
        is_skill_load_tool = tool_name == LOAD_SKILL_TOOL_NAME
        invocation_payload = self._tool_invocation_payload(
            arguments=arguments,
            is_shell_tool=is_shell_tool,
        )
        if is_shell_tool:
            report.shell_calls_total += 1
        started = time.monotonic()

        try:
            await self._bus.emit_simple(
                EV_TOOL_PREFLIGHT_START,
                **event_context_payload(
                    context,
                    run_kind=run_kind,
                    phase="tool_use",
                    scope_id=f"tool:{tool_call_id}",
                    correlation_id=tool_call_id,
                    source_component="tool_dispatch.preflight",
                ),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                **invocation_payload,
                is_shell_tool=is_shell_tool,
                is_skill_load_tool=is_skill_load_tool,
            )
            preflight_decision, preflight_source = await self.evaluate_tool_preflight(
                tool_name=tool_name,
                arguments=arguments,
                context=context,
                report=report,
                shell_capability=shell_capability,
                shell_tool_name=shell_tool_name,
            )
            pending_shell_request = None
            pending_shell_plan = None
            auto_approval_rule = None
            if (
                preflight_decision == PolicyDecision.CONFIRM
                and is_shell_tool
                and shell_capability is not None
            ):
                pending_shell_request = self._shell_handler.normalize_shell_request(
                    arguments,
                    shell_capability,
                    context.tool_context,
                )
                pending_shell_plan = self._shell_handler.build_shell_command_plan(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    request=pending_shell_request,
                    capability=shell_capability,
                )
                auto_approval_rule = self._shell_handler.find_matching_approval_rule(
                    context,
                    pending_shell_plan,
                )
                if auto_approval_rule is not None:
                    preflight_decision = PolicyDecision.ALLOW
                    preflight_source = "shell_approval_rule"
            await self._bus.emit_simple(
                EV_TOOL_PREFLIGHT_END,
                **event_context_payload(
                    context,
                    run_kind=run_kind,
                    phase="tool_use",
                    scope_id=f"tool:{tool_call_id}",
                    correlation_id=tool_call_id,
                    source_component="tool_dispatch.preflight",
                ),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                decision=(
                    preflight_decision.value if preflight_decision is not None else None
                ),
                source=preflight_source,
                requires_confirmation=preflight_decision == PolicyDecision.CONFIRM,
            )
            if auto_approval_rule is not None and pending_shell_plan is not None:
                await self._bus.emit_simple(
                    EV_TOOL_APPROVAL_RESOLVED,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch.approval",
                    ),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    plan_id=pending_shell_plan.plan_id,
                    decision="approved",
                    source="shell_approval_rule",
                    rule_id=auto_approval_rule.rule_id,
                )
            if preflight_decision == PolicyDecision.DENY:
                if is_shell_tool:
                    report.shell_calls_denied += 1
                denial_source = preflight_source or "policy"
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    content=f"[DENIED by {denial_source} preflight: {tool_name}]",
                    is_error=True,
                    metadata={
                        "safety_preflight_decision": "deny",
                        "safety_preflight_source": denial_source,
                    },
                )
            if preflight_decision == PolicyDecision.CONFIRM:
                if is_shell_tool:
                    report.shell_calls_confirm_required += 1
                    if shell_capability is None:
                        return ToolResult(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            content=f"[DENIED by shell capability preflight: {tool_name}]",
                            is_error=True,
                            metadata={
                                "safety_preflight_decision": "deny",
                                "safety_preflight_source": "shell_capability",
                            },
                        )
                    request = (
                        pending_shell_request
                        if pending_shell_request is not None
                        else self._shell_handler.normalize_shell_request(
                            arguments,
                            shell_capability,
                            context.tool_context,
                        )
                    )
                    pending_plan = (
                        pending_shell_plan
                        if pending_shell_plan is not None
                        else self._shell_handler.build_shell_command_plan(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            request=request,
                            capability=shell_capability,
                        )
                    )
                    await self._bus.emit_simple(
                        EV_TOOL_APPROVAL_REQUIRED,
                        **event_context_payload(
                            context,
                            run_kind=run_kind,
                            phase="tool_use",
                            scope_id=f"tool:{tool_call_id}",
                            correlation_id=tool_call_id,
                            source_component="tool_dispatch.approval",
                        ),
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        plan_id=pending_plan.plan_id,
                        command_preview=string_preview(
                            pending_plan.command,
                            limit=300,
                        ),
                        **invocation_payload,
                    )
                    raise PendingShellApprovalError(pending_plan)
                confirmation_source = preflight_source or "policy"
                await self._bus.emit_simple(
                    EV_TOOL_APPROVAL_REQUIRED,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch.approval",
                    ),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    plan_id=None,
                    command_preview=string_preview(arguments, limit=300),
                    **invocation_payload,
                )
                approved = self.confirm_destructive_action(
                    tool_name=tool_name,
                    arguments=arguments,
                    context=context,
                )
                report.destructive_action_requested += 1
                await self._bus.emit_simple(
                    EV_DESTRUCTIVE_ACTION,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch.approval",
                    ),
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=redact_sensitive_keys(arguments),
                    confirmed=approved,
                    source=confirmation_source,
                    decision=PolicyDecision.CONFIRM.value,
                )
                await self._bus.emit_simple(
                    EV_TOOL_APPROVAL_RESOLVED,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch.approval",
                    ),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    decision="approved" if approved else "rejected",
                )
                if approved:
                    report.destructive_action_confirmed += 1
                else:
                    return ToolResult(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        content=f"[REQUIRES CONFIRMATION: {tool_name} not confirmed]",
                        is_error=False,
                        metadata={
                            "safety_preflight_decision": "confirm_required",
                            "safety_preflight_source": confirmation_source,
                        },
                    )

            await self._bus.emit_simple(
                EV_TOOL_CALL_START,
                **event_context_payload(
                    context,
                    run_kind=run_kind,
                    phase="tool_use",
                    scope_id=f"tool:{tool_call_id}",
                    correlation_id=tool_call_id,
                    source_component="tool_dispatch",
                ),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                **invocation_payload,
                preflight_decision=(
                    preflight_decision.value if preflight_decision is not None else None
                ),
                preflight_source=preflight_source,
                is_shell_tool=is_shell_tool,
                is_skill_load_tool=is_skill_load_tool,
            )
            if is_skill_load_tool:
                await self._bus.emit_simple(
                    EV_SKILL_LOAD_START,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch",
                    ),
                    skill_name=arguments.get("name"),
                )

            # Path validation runs for ALL non-shell tools regardless of dispatch
            # path. Registry dispatch uses validate_paths=False because this gate
            # already performed the check.
            should_validate_paths = False
            if not is_shell_tool:
                if self._tool_registry and "fs" in self._tool_registry.get_tags(tool_name):
                    should_validate_paths = True
                if registry_definition is not None and registry_definition.filesystem_access:
                    should_validate_paths = True
                if config_definition is not None and config_definition.filesystem_access:
                    should_validate_paths = True
                if not should_validate_paths:
                    should_validate_paths = contains_path_argument(arguments)
            path_keys = None
            authoritative_definition = registry_definition or config_definition
            if authoritative_definition is not None and authoritative_definition.path_fields:
                path_keys = set(authoritative_definition.path_fields)
            if should_validate_paths:
                await self._bus.emit_simple(
                    EV_TOOL_VALIDATION_START,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch.validation",
                    ),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                )
                validate_path_arguments(arguments, context.tool_context, path_keys=path_keys)
                redacted_arguments = redact_sensitive_keys(arguments)
                redacted_keys: list[str] = []
                if isinstance(redacted_arguments, dict):
                    redacted_keys = sorted(
                        key for key, value in redacted_arguments.items() if value == "[REDACTED]"
                    )
                await self._bus.emit_simple(
                    EV_TOOL_VALIDATION_END,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch.validation",
                    ),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    ok=True,
                    validated_paths=sorted(path_keys) if path_keys is not None else sorted(arguments.keys()),
                    redacted_keys=redacted_keys,
                )

            result: ToolResult | None = None
            if is_shell_tool and shell_capability is not None and self._shell_executor is not None:
                await self._bus.emit_simple(
                    EV_TOOL_DISPATCH_SELECTED,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch",
                    ),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    dispatch_path="shell",
                    retryable=False,
                    handler_name=type(self._shell_executor).__name__,
                    **invocation_payload,
                )
                await self._bus.emit_simple(
                    EV_TOOL_EXECUTION_START,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch",
                    ),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    dispatch_path="shell",
                    **invocation_payload,
                )
                request = self._shell_handler.normalize_shell_request(
                    arguments,
                    shell_capability,
                    context.tool_context,
                )
                shell_executor = self._shell_executor
                async def _execute_shell(
                    request: Any = request,
                    tool_context: Any = context.tool_context,
                    capability: Any = shell_capability,
                ) -> Any:
                    return await shell_executor.execute(
                        request,
                        tool_context,
                        capability,
                    )
                shell_result = await self._policy_runner.call(
                    operation="tool.shell_execute",
                    report=report,
                    retryable=False,
                    fn=_execute_shell,
                )
                result = self._shell_handler.shell_result_to_tool_result(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    request=request,
                    result=shell_result,
                    capability=shell_capability,
                )
            elif self._tool_registry and tool_name in self._tool_registry:
                await self._bus.emit_simple(
                    EV_TOOL_DISPATCH_SELECTED,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch",
                    ),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    dispatch_path="registry",
                    retryable=self._is_tool_retryable(authoritative_definition),
                    handler_name=type(self._tool_registry).__name__,
                    **invocation_payload,
                )
                await self._bus.emit_simple(
                    EV_TOOL_EXECUTION_START,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch",
                    ),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    dispatch_path="registry",
                    **invocation_payload,
                )
                registry = self._tool_registry
                async def _dispatch_from_registry(
                    reg: ToolRegistry = registry,
                    name: str = tool_name,
                    args: dict[str, Any] = dict(arguments),
                    tool_context: Any = context.tool_context,
                    call_id: str = tool_call_id,
                ) -> ToolResult | None:
                    return await reg.dispatch(
                        name,
                        args,
                        tool_context,
                        validate_paths=False,
                        tool_call_id=call_id,
                    )
                result = await self._policy_runner.call(
                    operation="tool.registry_dispatch",
                    report=report,
                    retryable=self._is_tool_retryable(authoritative_definition),
                    fn=_dispatch_from_registry,
                )

            if result is None and self._tool_executor is not None:
                await self._bus.emit_simple(
                    EV_TOOL_DISPATCH_SELECTED,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch",
                    ),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    dispatch_path="executor",
                    retryable=self._is_tool_retryable(authoritative_definition),
                    handler_name=type(self._tool_executor).__name__,
                    **invocation_payload,
                )
                await self._bus.emit_simple(
                    EV_TOOL_EXECUTION_START,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch",
                    ),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    dispatch_path="executor",
                    **invocation_payload,
                )
                executor = self._tool_executor
                async def _execute_tool(
                    exec: ToolExecutor = executor,
                    name: str = tool_name,
                    args: dict[str, Any] = dict(arguments),
                    tool_context: Any = context.tool_context,
                ) -> ToolResult | None:
                    return await exec.execute(
                        name,
                        args,
                        tool_context,
                    )
                result = await self._policy_runner.call(
                    operation="tool.executor_execute",
                    report=report,
                    retryable=self._is_tool_retryable(authoritative_definition),
                    fn=_execute_tool,
                )
            if result is not None and not isinstance(result, ToolResult):
                raise TypeError(
                    f"dispatcher path returned {type(result).__name__}, expected ToolResult"
                )
        except PendingShellApprovalError:
            raise
        except (PathIsolationError, PermissionError, ValueError) as exc:
            await self._bus.emit_simple(
                EV_TOOL_CALL_FAILED,
                **event_context_payload(
                    context,
                    run_kind=run_kind,
                    phase="tool_use",
                    scope_id=f"tool:{tool_call_id}",
                    correlation_id=tool_call_id,
                    source_component="tool_dispatch",
                ),
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                error_code=type(exc).__name__,
                error_message=str(exc),
                **invocation_payload,
            )
            result = self._tool_error_result(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                category="validation",
                exc=exc,
            )
        except Exception as exc:
            logger.exception("Tool dispatch failed: tool=%s", tool_name)
            await self._bus.emit_simple(
                EV_TOOL_CALL_FAILED,
                **event_context_payload(
                    context,
                    run_kind=run_kind,
                    phase="tool_use",
                    scope_id=f"tool:{tool_call_id}",
                    correlation_id=tool_call_id,
                    source_component="tool_dispatch",
                ),
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                error_code=type(exc).__name__,
                error_message=str(exc),
                **invocation_payload,
            )
            result = self._tool_error_result(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                category="execution",
                exc=exc,
            )

        if result is None:
            result = ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                content=f"[ERROR: tool '{tool_name}' not found]",
                is_error=True,
            )

        latency_ms = (time.monotonic() - started) * 1000
        result = result.model_copy(
            update={
                "tool_call_id": tool_call_id or result.tool_call_id,
                "tool_name": tool_name or result.tool_name,
                "latency_ms": latency_ms,
            }
        )
        report.tool_latency_ms.append(latency_ms)
        if is_shell_tool or matches_shell_tool_name:
            for flag in result.metadata.get(ToolResultMeta.SHELL_RISK_FLAGS, []):
                if isinstance(flag, str) and flag not in report.shell_risk_flags:
                    report.shell_risk_flags.append(flag)
        if result.prompt_injection_signal:
            await self._bus.emit_simple(
                EV_INJECTION_SIGNAL,
                **event_context_payload(
                    context,
                    run_kind=run_kind,
                    phase="tool_use",
                    scope_id=f"tool:{tool_call_id}",
                    correlation_id=tool_call_id,
                    source_component="tool_dispatch",
                ),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                result_summary=tool_payload_summary(result),
            )

        if is_skill_load_tool:
            skill_name = result.metadata.get(ToolResultMeta.SKILL_NAME) or arguments.get(
                "name"
            )
            if result.metadata.get(ToolResultMeta.SKILL_BUDGET_EXCEEDED) is True:
                report.add_warning(ToolResultMeta.SKILL_BUDGET_EXCEEDED)
                await self._bus.emit_simple(
                    EV_SKILL_BUDGET_EXCEEDED,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch",
                    ),
                    skill_name=skill_name,
                    load_count=result.metadata.get(ToolResultMeta.SKILL_LOAD_COUNT),
                    max_loads=result.metadata.get(
                        ToolResultMeta.MAX_SKILL_LOADS_PER_RUN
                    ),
                )
            else:
                if (
                    isinstance(skill_name, str)
                    and skill_name
                    and f"skill_loaded:{skill_name}" not in report.artifacts
                ):
                    report.add_artifact(f"skill_loaded:{skill_name}")
                await self._bus.emit_simple(
                    EV_SKILL_LOAD_END,
                    **event_context_payload(
                        context,
                        run_kind=run_kind,
                        phase="tool_use",
                        scope_id=f"tool:{tool_call_id}",
                        correlation_id=tool_call_id,
                        source_component="tool_dispatch",
                    ),
                    skill_name=skill_name,
                    is_error=result.is_error,
                    from_cache=result.metadata.get(
                        ToolResultMeta.SKILL_FROM_CACHE,
                        False,
                    ),
                    truncated=result.metadata.get(
                        ToolResultMeta.SKILL_TRUNCATED,
                        False,
                    ),
                )

        await self._bus.emit_simple(
            EV_TOOL_EXECUTION_END,
            **event_context_payload(
                context,
                run_kind=run_kind,
                phase="tool_use",
                scope_id=f"tool:{tool_call_id}",
                correlation_id=tool_call_id,
                source_component="tool_dispatch",
            ),
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            latency_ms=latency_ms,
            exit_code=result.metadata.get("exit_code"),
            is_error=result.is_error,
            **invocation_payload,
        )
        await self._bus.emit_simple(
            EV_TOOL_RESULT_READY,
            **event_context_payload(
                context,
                run_kind=run_kind,
                phase="tool_use",
                scope_id=f"tool:{tool_call_id}",
                correlation_id=tool_call_id,
                source_component="tool_dispatch",
            ),
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result_preview=string_preview(result.content, limit=600),
            artifact_summary=result.artifact_summary,
            artifact_path=result.artifact_path,
            is_error=result.is_error,
            prompt_injection_signal=result.prompt_injection_signal,
            **invocation_payload,
        )
        await self._bus.emit_simple(
            EV_TOOL_CALL_END,
            **event_context_payload(
                context,
                run_kind=run_kind,
                phase="tool_use",
                scope_id=f"tool:{tool_call_id}",
                correlation_id=tool_call_id,
                source_component="tool_dispatch",
            ),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            latency_ms=latency_ms,
            is_error=result.is_error,
            prompt_injection_signal=result.prompt_injection_signal,
            result=tool_payload_summary(result),
            **invocation_payload,
        )
        try:
            self._hooks.call_tool_post_execute(
                result=result,
                context=context.tool_context,
                report=report,
            )
        except Exception as exc:
            logger.exception("tool_post_execute hook failed: tool=%s", tool_name)
            report.add_warning(f"tool_post_execute_hook_failed:{tool_name}:{type(exc).__name__}")
        return result

    async def evaluate_tool_preflight(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        context: AgentContext,
        report: ExecutionReport,
        shell_capability: ShellToolConfig | None,
        shell_tool_name: str,
    ) -> tuple[PolicyDecision | None, str | None]:
        visible_tool_names_raw = context.tool_context.metadata.get(
            ToolContextMeta.VISIBLE_TOOL_NAMES
        )
        if ToolContextMeta.VISIBLE_TOOL_NAMES in context.tool_context.metadata and isinstance(
            visible_tool_names_raw, list
        ):
            visible_tool_names = {
                name for name in visible_tool_names_raw if isinstance(name, str)
            }
            if tool_name not in visible_tool_names:
                return PolicyDecision.DENY, "tool_visibility"

        hook_decision = normalize_policy_decision(
            self._hooks.call_tool_pre_execute(
                tool_name=tool_name,
                arguments=arguments,
                context=context.tool_context,
                report=report,
            )
        )
        policy_decision: PolicyDecision | None = None
        if self._policy is not None:
            policy_decision = normalize_policy_decision(
                await self._policy.evaluate(tool_name, arguments, context.tool_context)
            )

        shell_decision: PolicyDecision | None = None
        shell_source: str | None = None
        if tool_name == shell_tool_name:
            if shell_capability is None:
                shell_decision = PolicyDecision.DENY
                shell_source = "shell_capability"
            elif self._shell_executor is None:
                shell_decision = PolicyDecision.DENY
                shell_source = "shell_executor"
            else:
                try:
                    request = self._shell_handler.normalize_shell_request(
                        arguments,
                        shell_capability,
                        context.tool_context,
                    )
                except Exception as exc:
                    report.add_warning(
                        f"shell_request_invalid:{type(exc).__name__}"
                    )
                    shell_decision = PolicyDecision.DENY
                    shell_source = "shell_contract"
                else:
                    if shell_capability.safety_mode.value == "yolo":
                        shell_decision = PolicyDecision.ALLOW
                        shell_source = "shell_yolo"
                    elif self._shell_safety_policy is not None:
                        shell_decision = normalize_policy_decision(
                            await self._shell_safety_policy.evaluate(
                                request,
                                context.tool_context,
                                shell_capability,
                            )
                        )
                        if shell_decision is not None:
                            shell_source = "shell_policy"

        for decision in (
            PolicyDecision.DENY,
            PolicyDecision.CONFIRM,
            PolicyDecision.ALLOW,
        ):
            sources: list[str] = []
            if hook_decision == decision:
                sources.append("hook")
            if policy_decision == decision:
                sources.append("policy")
            if shell_decision == decision:
                sources.append(shell_source or "shell")
            if sources:
                return decision, "+".join(sources)
        return None, None

    def confirm_destructive_action(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        context: AgentContext,
    ) -> bool:
        approved_result = self._hooks.call_destructive_action(
            tool_name=tool_name,
            arguments=arguments,
            context=context.tool_context,
        )
        return approved_result is True

    def _normalize_tool_call(
        self,
        *,
        tool_call: Any,
        report: ExecutionReport,
    ) -> tuple[str, str, Any] | ToolResult:
        if isinstance(tool_call, ToolCall):
            tool_payload = tool_call.to_openai_dict()
        else:
            tool_payload = tool_call
        if not isinstance(tool_payload, dict):
            report.add_warning("tool_call_malformed:payload_type")
            logger.warning("Malformed tool_call payload type: %r", type(tool_call).__name__)
            return ToolResult(
                tool_call_id="",
                tool_name="unknown",
                content="[MALFORMED TOOL CALL: payload must be an object]",
                is_error=True,
                metadata={"tool_call_malformed": True},
            )
        func_data = tool_payload.get("function", tool_payload)
        if not isinstance(func_data, dict):
            report.add_warning("tool_call_malformed:function_payload")
            logger.warning("Malformed tool_call function payload: %r", tool_payload)
            return ToolResult(
                tool_call_id=str(tool_payload.get("id", "")),
                tool_name="unknown",
                content="[MALFORMED TOOL CALL: function payload must be an object]",
                is_error=True,
                metadata={"tool_call_malformed": True},
            )
        tool_name = func_data.get("name", "")
        if not isinstance(tool_name, str) or not tool_name.strip():
            report.add_warning("tool_call_malformed:name")
            logger.warning("Malformed tool_call missing name: %r", tool_payload)
            return ToolResult(
                tool_call_id=str(tool_payload.get("id", "")),
                tool_name="unknown",
                content="[MALFORMED TOOL CALL: function.name must be a non-empty string]",
                is_error=True,
                metadata={"tool_call_malformed": True},
            )
        raw_arguments = func_data.get("arguments", "{}")
        if not isinstance(raw_arguments, (str, dict)):
            report.add_warning(f"tool_arguments_invalid_type:{tool_name}")
            logger.warning(
                "Malformed tool_call arguments type: tool=%s type=%s",
                tool_name,
                type(raw_arguments).__name__,
            )
            return ToolResult(
                tool_call_id=str(tool_payload.get("id", "")),
                tool_name=tool_name,
                content="[MALFORMED TOOL CALL: function.arguments must be a JSON string or object]",
                is_error=True,
                metadata={"tool_call_malformed": True},
            )
        return str(tool_payload.get("id", "")), tool_name, raw_arguments

    @staticmethod
    def _parse_tool_arguments(
        *,
        tool_name: str,
        raw_args: Any,
        report: ExecutionReport,
    ) -> dict[str, Any]:
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except Exception as exc:
                logger.warning(
                    "Failed to parse tool arguments as JSON: tool=%s error=%s raw=%.200s",
                    tool_name,
                    type(exc).__name__,
                    raw_args,
                )
                report.add_warning(f"tool_arguments_parse_failed:{tool_name}")
                raise ValueError(
                    f"tool={tool_name} arguments are not valid JSON"
                ) from exc
            if not isinstance(parsed, dict):
                report.add_warning(f"tool_arguments_not_object:{tool_name}")
                raise ValueError(
                    f"tool={tool_name} arguments must decode to a JSON object"
                )
            return parsed
        if isinstance(raw_args, dict):
            return raw_args
        report.add_warning(f"tool_arguments_invalid_type:{tool_name}")
        raise ValueError(
            f"tool={tool_name} arguments must be provided as a JSON object or string"
        )

    @staticmethod
    def _is_tool_retryable(definition: Any) -> bool:
        return bool(getattr(definition, "idempotent", False))

    @staticmethod
    def _tool_error_result(
        *,
        tool_call_id: str,
        tool_name: str,
        category: str,
        exc: Exception,
    ) -> ToolResult:
        category_label = "VALIDATION" if category == "validation" else "EXECUTION"
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=f"[TOOL {category_label} ERROR: {type(exc).__name__}: {exc}]",
            is_error=True,
            metadata={
                "error_category": category,
                "error_type": type(exc).__name__,
            },
        )


__all__ = ["ToolDispatcher"]
