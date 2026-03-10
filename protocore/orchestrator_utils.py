"""Shared helpers extracted from `orchestrator.py`."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal, TypeVar

from .constants import (
    MAX_EXTRA_BODY_MERGE_DEPTH,
    MAX_PARALLEL_POLICY_CONCURRENCY,
    MAX_RECOVERY_TEXT_SIZE,
    MAX_RESULT_ERROR_MESSAGE_LENGTH,
    MAX_RETRY_ATTEMPTS,
    SUPPORTED_PARALLEL_CANCELLATION_MODES,
)
from .orchestrator_errors import ContractViolationError
from .protocols import ParallelExecutionPolicy, RetryPolicy, TimeoutPolicy
from .types import (
    AgentConfig,
    AgentContext,
    AgentContextMeta,
    ExecutionReport,
    ExecutionStatus,
    Message,
    PlanArtifact,
    PolicyDecision,
    ShellToolConfig,
    StopReason,
    SubagentResult,
    SubagentRunSummary,
    SubagentStatus,
    RunKind,
    ToolCallRecord,
    ToolContextMeta,
    ToolDefinition,
    ToolResult,
    ToolResultMeta,
)

logger = logging.getLogger(__name__)

LLMCallPurpose = Literal["main_loop", "finalize", "auto_compact", "manual_compact"]
MessageSerializationTarget = Literal["chat", "responses"]


def _system_content_to_string(content: Any) -> str:
    """Extract plain text from a message content for merging into system block."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip()
    parts: list[str] = []
    for part in content:
        part_type = getattr(part, "type", "text")
        if part_type == "text":
            text = getattr(part, "text", "") or ""
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        elif part_type == "input_json":
            parts.append(
                json.dumps(getattr(part, "json_data", None) or {}, ensure_ascii=True)
            )
    return "\n".join(parts)


def _should_skip_embedded_system_message(
    message: Message,
    *,
    system: str | None,
    duplicate_already_skipped: bool,
) -> bool:
    """Used only when building merged system parts; skip exact duplicate of system param."""
    if (
        message.role != "system"
        or system is None
        or duplicate_already_skipped
        or not isinstance(message.content, str)
    ):
        return False
    return message.content.strip() == system.strip()


def _serialize_content_for_target(
    content: Any,
    *,
    target_api: MessageSerializationTarget,
) -> str | list[dict[str, Any]]:
    if content is None:
        return ""
    if isinstance(content, str):
        if target_api == "responses":
            return [{"type": "input_text", "text": content}]
        return content
    if not isinstance(content, list):
        return str(content)

    converted: list[dict[str, Any]] = []
    for part in content:
        part_type = getattr(part, "type", "text")
        if part_type == "text":
            text = getattr(part, "text", "") or ""
            if target_api == "responses":
                converted.append({"type": "input_text", "text": text})
            else:
                converted.append({"type": "text", "text": text})
            continue
        if part_type == "image_url":
            image_url = getattr(part, "image_url", None) or {}
            if target_api == "responses":
                payload = {
                    "type": "input_image",
                    "image_url": image_url.get("url", ""),
                }
                detail = image_url.get("detail")
                if detail:
                    payload["detail"] = detail
                converted.append(payload)
            else:
                converted.append({"type": "image_url", "image_url": image_url})
            continue
        if part_type == "input_json":
            json_text = json.dumps(
                getattr(part, "json_data", None) or {},
                ensure_ascii=True,
                sort_keys=True,
            )
            text_payload = (
                {"type": "input_text", "text": json_text}
                if target_api == "responses"
                else {"type": "text", "text": json_text}
            )
            converted.append(text_payload)
            continue
        converted.append({"type": str(part_type)})
    return converted


def _serialize_function_output(
    content: Any,
) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    fragments: list[str] = []
    if not isinstance(content, list):
        return str(content)
    for part in content:
        part_type = getattr(part, "type", "text")
        if part_type == "text":
            text = getattr(part, "text", None)
            if text:
                fragments.append(text)
        elif part_type == "input_json":
            fragments.append(
                json.dumps(getattr(part, "json_data", None) or {}, ensure_ascii=True)
            )
        elif part_type == "image_url":
            image_url = getattr(part, "image_url", None) or {}
            fragments.append(image_url.get("url", ""))
    return "\n".join(fragment for fragment in fragments if fragment)


def serialize_messages_for_api(
    messages: list[Message],
    *,
    system: str | None,
    target_api: MessageSerializationTarget,
) -> list[dict[str, Any]]:
    """Serialize internal Message models into chat/responses request payloads.

    System message is always a single message at the beginning: all system content
    (from the ``system`` argument and from any ``Message(role="system", ...)`` in
    ``messages``) is merged into one block so backends that require "system first
    and only once" (e.g. Qwen/vLLM) are satisfied.
    """
    system_parts: list[str] = []
    if system and system.strip():
        system_parts.append(system.strip())
    skipped_embedded_system = False
    for msg in messages:
        if msg.role != "system":
            continue
        if _should_skip_embedded_system_message(
            msg,
            system=system,
            duplicate_already_skipped=skipped_embedded_system,
        ):
            skipped_embedded_system = True
            continue
        text = _system_content_to_string(msg.content)
        if text:
            system_parts.append(text)

    single_system_content: str | list[dict[str, Any]]
    if system_parts:
        merged_system_text = "\n\n".join(system_parts)
        if target_api == "responses":
            single_system_content = [{"type": "input_text", "text": merged_system_text}]
        else:
            single_system_content = merged_system_text
    else:
        single_system_content = "" if target_api == "chat" else []

    serialized: list[dict[str, Any]] = []
    if system_parts:
        serialized.append({"role": "system", "content": single_system_content})

    for msg in messages:
        if msg.role == "system":
            continue

        tool_calls = msg.tool_calls or []
        tool_call_payloads = [
            tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else tc
            for tc in tool_calls
        ]

        if target_api == "responses" and msg.role == "tool":
            serialized.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id or "",
                    "output": _serialize_function_output(msg.content),
                }
            )
            continue

        if target_api == "responses" and msg.role == "assistant" and tool_call_payloads:
            if msg.content not in (None, "", []):
                serialized.append(
                    {
                        "role": "assistant",
                        "content": _serialize_content_for_target(
                            msg.content,
                            target_api=target_api,
                        ),
                    }
                )
            for tool_call_payload in tool_call_payloads:
                if not isinstance(tool_call_payload, dict):
                    continue
                function_data = tool_call_payload.get("function", tool_call_payload)
                if not isinstance(function_data, dict):
                    continue
                serialized.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call_payload.get("id", ""),
                        "name": function_data.get("name", ""),
                        "arguments": function_data.get("arguments", "{}"),
                    }
                )
            continue

        item: dict[str, Any] = {
            "role": msg.role,
            "content": _serialize_content_for_target(
                msg.content,
                target_api=target_api,
            ),
        }
        if tool_call_payloads:
            item["tool_calls"] = tool_call_payloads
        if msg.tool_call_id:
            item["tool_call_id"] = msg.tool_call_id
        if msg.name:
            item["name"] = msg.name
        serialized.append(item)
    return serialized


def resolve_effective_llm_config(
    config: AgentConfig | None,
    *,
    run_kind: RunKind = RunKind.LEADER,
    call_purpose: LLMCallPurpose = "main_loop",
) -> AgentConfig:
    """Resolve config defaults for a specific LLM call site.

    ``run_kind`` and ``call_purpose`` are part of the contract so all call
    sites share one source of truth even though the current selective-thinking
    logic remains purpose-agnostic. They are reserved for future per-run
    overrides such as compaction-only models or subagent-specific sampling.
    """
    if config is None:
        raise ValueError("config is required to resolve effective LLM settings")
    _ = run_kind, call_purpose
    return config.resolved_with_selective_thinking()


def resolve_max_tokens(config: AgentConfig) -> int | None:
    if config.max_tokens is None:
        return None
    return config.max_tokens + max(config.thinking_tokens_reserve, 0)


def merge_nested_dict(
    base: dict[str, Any],
    overlay: dict[str, Any],
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> dict[str, Any]:
    merged = dict(base)
    seen = set() if _seen is None else _seen
    current_ids = {id(base), id(overlay)}
    if _depth >= MAX_EXTRA_BODY_MERGE_DEPTH or current_ids & seen:
        return overlay if _depth > 0 else {**base, **overlay}
    seen.update(current_ids)
    for key, value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = merge_nested_dict(
                base_value,
                value,
                _depth=_depth + 1,
                _seen=seen,
            )
        else:
            merged[key] = value
    return merged


def build_llm_request_kwargs(
    config: AgentConfig,
    *,
    has_tools: bool = False,
) -> dict[str, Any]:
    request_kwargs = dict(config.llm_request_kwargs)

    if config.top_p is not None:
        request_kwargs.setdefault("top_p", config.top_p)
    if config.presence_penalty is not None:
        request_kwargs.setdefault("presence_penalty", config.presence_penalty)

    extra_body: dict[str, Any] = dict(config.llm_extra_body)
    from_kwargs_extra = request_kwargs.pop("extra_body", None)
    if isinstance(from_kwargs_extra, dict):
        extra_body = merge_nested_dict(extra_body, from_kwargs_extra)

    if config.top_k is not None:
        extra_body.setdefault("top_k", config.top_k)
    if config.min_p is not None:
        extra_body.setdefault("min_p", config.min_p)
    if config.repetition_penalty is not None:
        extra_body.setdefault("repetition_penalty", config.repetition_penalty)

    if config.enable_thinking is not None:
        if "enable_thinking" not in extra_body:
            chat_template_kwargs = extra_body.get("chat_template_kwargs")
            if not isinstance(chat_template_kwargs, dict):
                chat_template_kwargs = {}
            chat_template_kwargs.setdefault("enable_thinking", config.enable_thinking)
            extra_body["chat_template_kwargs"] = chat_template_kwargs

    if has_tools:
        request_kwargs.setdefault("parallel_tool_calls", config.parallel_tool_calls)

    if extra_body:
        # Emit explicit debug payload to verify extra_body passthrough.
        logger.debug(
            "LLM request extra_body prepared: agent=%s has_tools=%s keys=%s payload=%s",
            config.agent_id,
            has_tools,
            sorted(extra_body.keys()),
            extra_body,
        )
        request_kwargs["extra_body"] = extra_body

    return request_kwargs


def extract_task(
    messages: list[Message],
    *,
    strategy: Literal["first", "last", "all"] = "all",
) -> str | None:
    """Extract a normalized user task from conversation history.

    The default ``all`` strategy preserves follow-up clarifications instead of
    silently taking only the first user turn.
    """

    def _message_text(msg: Message) -> str:
        content = msg.content or ""
        if isinstance(content, list):
            return " ".join(getattr(part, "text", "") or "" for part in content).strip()
        return str(content).strip()

    user_texts = [
        text
        for msg in messages
        if msg.role == "user"
        for text in [_message_text(msg)]
        if text
    ]
    if not user_texts:
        return None
    if strategy == "first":
        return user_texts[0]
    if strategy == "last":
        return user_texts[-1]
    return "\n\n".join(user_texts)


def build_subagent_summary(report: ExecutionReport) -> SubagentRunSummary:
    finished_at = report.finished_at or datetime.now(timezone.utc).isoformat()
    return SubagentRunSummary(
        agent_id=report.agent_id,
        status=report.status,
        started_at=report.started_at,
        finished_at=finished_at,
        duration_ms=report.duration_ms or 0.0,
        tool_calls_total=report.tool_calls_total,
        input_tokens=report.input_tokens,
        output_tokens=report.output_tokens,
        errors=[report.error_message] if report.error_message else [],
    )


def normalize_policy_decision(decision: object | None) -> PolicyDecision | None:
    if decision is None:
        return None
    if isinstance(decision, (list, tuple)):
        for item in decision:
            normalized = normalize_policy_decision(item)
            if normalized is not None:
                return normalized
        return None
    if isinstance(decision, PolicyDecision):
        return decision
    if isinstance(decision, str):
        try:
            return PolicyDecision(decision.strip().lower())
        except ValueError:
            return None
    return None


def session_refs(context: AgentContext) -> tuple[str, str]:
    message_history_ref = context.message_history_ref or f"session:{context.session_id}:messages"
    execution_metadata_ref = (
        context.execution_metadata_ref or f"request:{context.request_id}:metadata"
    )
    return message_history_ref, execution_metadata_ref


def load_existing_plan(context: AgentContext) -> PlanArtifact | None:
    raw_plan = context.metadata.get(AgentContextMeta.PLAN_ARTIFACT)
    if raw_plan is None:
        return None
    if isinstance(raw_plan, PlanArtifact):
        return raw_plan
    if isinstance(raw_plan, dict):
        return PlanArtifact.model_validate(raw_plan)
    return None


def default_stop_reason_for_status(status: ExecutionStatus) -> StopReason:
    if status == ExecutionStatus.COMPLETED:
        return StopReason.END_TURN
    if status == ExecutionStatus.PARTIAL:
        return StopReason.END_TURN
    if status == ExecutionStatus.CANCELLED:
        return StopReason.CANCELLED
    return StopReason.ERROR


def ensure_terminal_report(report: ExecutionReport, result: Any) -> None:
    """Normalize workflow/external reports into a terminal execution report."""
    if report.status != ExecutionStatus.RUNNING and report.finished_at is not None:
        return

    status = result.status
    if status == ExecutionStatus.RUNNING:
        logger.warning("Workflow returned RUNNING status; normalizing to COMPLETED")
        status = ExecutionStatus.COMPLETED

    error_message = report.error_message
    if error_message is None and status in {ExecutionStatus.PARTIAL, ExecutionStatus.FAILED}:
        if result.content:
            error_message = result.content[:MAX_RESULT_ERROR_MESSAGE_LENGTH]
        else:
            error_message = status.value

    report.finalize(
        status,
        stop_reason=report.stop_reason or default_stop_reason_for_status(status),
        error_code=report.error_code,
        error_message=error_message,
    )


def merge_execution_report(
    target: ExecutionReport,
    source: ExecutionReport,
    *,
    include_terminal: bool = True,
) -> None:
    target_is_terminal = (
        target.status != ExecutionStatus.RUNNING and target.finished_at is not None
    )
    if include_terminal and not target_is_terminal and source.status != ExecutionStatus.RUNNING:
        target.status = source.status
        target.stop_reason = source.stop_reason
        target.error_code = source.error_code
        target.error_message = source.error_message
        target.finished_at = source.finished_at
        target.duration_ms = source.duration_ms
    if source.queue_wait_ms is not None:
        target.queue_wait_ms = max(target.queue_wait_ms or 0.0, source.queue_wait_ms)
    target.loop_count += source.loop_count
    target.llm_latency_ms.extend(source.llm_latency_ms)
    target.tool_latency_ms.extend(source.tool_latency_ms)
    target.input_tokens += source.input_tokens
    target.output_tokens += source.output_tokens
    target.cached_tokens += source.cached_tokens
    target.reasoning_tokens += source.reasoning_tokens
    if source.estimated_cost is not None:
        target.estimated_cost = (target.estimated_cost or 0.0) + source.estimated_cost
    target.tool_calls_total += source.tool_calls_total
    for name, count in source.tool_calls_by_name.items():
        target.increment_tool_call(name, count)
    target.tool_failures += source.tool_failures
    target.forced_finalization_triggered = (
        target.forced_finalization_triggered or source.forced_finalization_triggered
    )
    target.state_manager_timeout_count += source.state_manager_timeout_count
    target.micro_compact_applied += source.micro_compact_applied
    target.auto_compact_applied += source.auto_compact_applied
    target.auto_compact_failed += source.auto_compact_failed
    target.manual_compact_applied += source.manual_compact_applied
    if source.tokens_before_compression_total is not None:
        target.tokens_before_compression_total = (
            (target.tokens_before_compression_total or 0)
            + source.tokens_before_compression_total
        )
    if source.tokens_after_compression_total is not None:
        target.tokens_after_compression_total = (
            (target.tokens_after_compression_total or 0)
            + source.tokens_after_compression_total
        )
    existing_compression_events = {
        (event.kind, event.timestamp) for event in target.compression_events
    }
    for event in source.compression_events:
        event_key = (event.kind, event.timestamp)
        if event_key not in existing_compression_events:
            target.compression_events.append(event)
            existing_compression_events.add(event_key)
    target.workflow_id = source.workflow_id or target.workflow_id
    target.node_count = (
        source.node_count if source.node_count is not None else target.node_count
    )
    target.edge_count = (
        source.edge_count if source.edge_count is not None else target.edge_count
    )
    target.node_durations_ms.update(source.node_durations_ms)
    target.plan_created = target.plan_created or source.plan_created
    target.plan_id = source.plan_id or target.plan_id
    target.plan_artifact = source.plan_artifact or target.plan_artifact
    target.subagents_parallel_max = max(
        target.subagents_parallel_max, source.subagents_parallel_max
    )
    target.destructive_action_requested += source.destructive_action_requested
    target.destructive_action_confirmed += source.destructive_action_confirmed
    target.prompt_injection_signals += source.prompt_injection_signals
    target.shell_calls_total += source.shell_calls_total
    target.shell_calls_denied += source.shell_calls_denied
    target.shell_calls_confirm_required += source.shell_calls_confirm_required
    target.shell_approvals_granted += source.shell_approvals_granted
    target.shell_approvals_rejected += source.shell_approvals_rejected
    existing_shell_risk_flags = set(target.shell_risk_flags)
    for flag in source.shell_risk_flags:
        if flag not in existing_shell_risk_flags:
            target.shell_risk_flags.append(flag)
            existing_shell_risk_flags.add(flag)
    target.artifacts_dropped += source.artifacts_dropped
    target.files_changed_dropped += source.files_changed_dropped
    target.artifacts_overflow = target.artifacts_overflow or source.artifacts_overflow
    target.files_changed_overflow = (
        target.files_changed_overflow or source.files_changed_overflow
    )
    existing_artifacts = set(target.artifacts)
    for item in source.artifacts:
        if item not in existing_artifacts:
            target.add_artifact(item)
            existing_artifacts.add(item)
    existing_files_changed = set(target.files_changed)
    for file_path in source.files_changed:
        if file_path not in existing_files_changed:
            target.add_file_changed(file_path)
            existing_files_changed.add(file_path)
    existing_warnings = set(target.warnings)
    for warning in source.warnings:
        if warning not in existing_warnings:
            target.add_warning(warning)
            existing_warnings.add(warning)
    existing_subagent_runs = {
        (run.agent_id, run.started_at) for run in target.subagent_runs
    }
    for run in source.subagent_runs:
        run_key = (run.agent_id, run.started_at)
        if run_key not in existing_subagent_runs:
            target.subagent_runs.append(run)
            existing_subagent_runs.add(run_key)
    existing_tool_call_details = {
        (detail.tool_call_id, detail.tool_name, detail.timestamp)
        for detail in target.tool_call_details
    }
    for detail in source.tool_call_details:
        detail_key = (detail.tool_call_id, detail.tool_name, detail.timestamp)
        if detail_key not in existing_tool_call_details:
            target.add_tool_call_detail(ToolCallRecord.model_validate(detail))
            existing_tool_call_details.add(detail_key)


def subagent_status_from_execution_status(status: ExecutionStatus) -> SubagentStatus:
    if status == ExecutionStatus.COMPLETED:
        return SubagentStatus.SUCCESS
    if status in {ExecutionStatus.CANCELLED, ExecutionStatus.PARTIAL}:
        return SubagentStatus.PARTIAL
    return SubagentStatus.FAILED


def execution_status_from_subagent_status(status: SubagentStatus) -> ExecutionStatus:
    if status == SubagentStatus.SUCCESS:
        return ExecutionStatus.COMPLETED
    if status == SubagentStatus.PARTIAL:
        return ExecutionStatus.PARTIAL
    return ExecutionStatus.FAILED


def append_tool_results_as_messages(
    messages: list[Message],
    tool_results: list[ToolResult],
) -> None:
    for tool_result in tool_results:
        content = tool_result.content
        prefixes: list[str] = []
        if tool_result.is_error:
            prefixes.append("[TOOL ERROR]")
        if tool_result.prompt_injection_signal:
            prefixes.append("[UNTRUSTED OUTPUT]")
        if prefixes:
            content = "\n".join([*prefixes, content])
        messages.append(
            Message(
                role="tool",
                content=content,
                tool_call_id=tool_result.tool_call_id,
                name=tool_result.tool_name,
            )
        )


def recover_tool_calls_from_assistant_text(
    content: str | list[Any] | None,
    runtime_tools: list[ToolDefinition],
    *,
    blocked_tool_names: set[str] | None = None,
    max_candidates: int = 5,
) -> list[dict[str, Any]]:
    """Parse tool calls from an assistant's plain-text response.

    WARNING: This recovery path must only be used on trusted assistant messages.
    Never run it on tool outputs or user content, otherwise external text could
    be reinterpreted as executable tool calls.
    """
    if not isinstance(content, str):
        return []

    text = content.strip()[:MAX_RECOVERY_TEXT_SIZE]
    if not text:
        return []

    blocked = {name for name in (blocked_tool_names or set()) if name}
    tool_names = {tool.name for tool in runtime_tools if tool.name and tool.name not in blocked}
    if not tool_names:
        return []

    candidates: list[str] = [text]
    fenced_blocks = re.findall(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    lowered_text = text.lower()
    for tag in ("tool_call", "function-call", "tool_response"):
        if f"</{tag}>" not in lowered_text:
            continue
        pattern = rf"<{tag}>(.*?)</{tag}>"
        tag_blocks = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(block.strip() for block in tag_blocks if block.strip())

    recovered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for candidate in candidates:
        if len(recovered) >= max_candidates:
            break
        recovered.extend(
            parse_json_tool_call_candidates(candidate, tool_names, seen, len(recovered))
        )
        if len(recovered) >= max_candidates:
            break
        recovered.extend(
            parse_legacy_tag_tool_call_candidates(candidate, tool_names, seen, len(recovered))
        )
        if len(recovered) >= max_candidates:
            break

    return recovered[:max_candidates]


def parse_json_tool_call_candidates(
    text: str,
    tool_names: set[str],
    seen: set[tuple[str, str]],
    offset: int,
) -> list[dict[str, Any]]:
    candidates: list[str] = [text.strip()]
    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and first_obj < last_obj:
        candidates.append(text[first_obj : last_obj + 1].strip())

    parsed: list[dict[str, Any]] = []
    for raw in candidates:
        if not raw or not (
            (raw.startswith("{") and raw.endswith("}"))
            or (raw.startswith("[") and raw.endswith("]"))
        ):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        payloads = payload if isinstance(payload, list) else [payload]
        for item in payloads:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name or name not in tool_names:
                continue
            arguments = item.get("arguments", {})
            if isinstance(arguments, str):
                normalized_args = arguments
            elif isinstance(arguments, dict):
                normalized_args = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
            else:
                normalized_args = "{}"
            fingerprint = (name, normalized_args)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            parsed.append(
                {
                    "id": f"fallback_{name}_{offset + len(parsed) + 1}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": normalized_args,
                    },
                }
            )
    return parsed


def parse_legacy_tag_tool_call_candidates(
    text: str,
    tool_names: set[str],
    seen: set[tuple[str, str]],
    offset: int,
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    if "</function>" not in text.lower():
        return parsed
    function_matches = re.finditer(
        r"<function\s*=\s*([a-zA-Z0-9_.:-]+)\s*>(.*?)</function>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for match in function_matches:
        name = match.group(1).strip()
        if not name or name not in tool_names:
            continue
        body = match.group(2)
        args: dict[str, Any] = {}
        if "</parameter>" not in body.lower():
            continue
        for param_match in re.finditer(
            r"<parameter\s*=\s*([a-zA-Z0-9_.:-]+)\s*>(.*?)</parameter>",
            body,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            key = param_match.group(1).strip()
            value = param_match.group(2).strip()
            if key:
                args[key] = value
        normalized_args = json.dumps(args, ensure_ascii=False) if args else "{}"
        fingerprint = (name, normalized_args)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        parsed.append(
            {
                "id": f"fallback_{name}_{offset + len(parsed) + 1}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": normalized_args,
                },
            }
        )
    return parsed


def validate_parallel_policy(policy: ParallelExecutionPolicy) -> None:
    if policy.max_concurrency <= 0:
        raise ContractViolationError(
            "PARALLEL_MAX_CONCURRENCY_INVALID",
            "ParallelExecutionPolicy.max_concurrency must be greater than zero",
        )
    if policy.max_concurrency > MAX_PARALLEL_POLICY_CONCURRENCY:
        raise ContractViolationError(
            "PARALLEL_MAX_CONCURRENCY_EXCESSIVE",
            (
                "ParallelExecutionPolicy.max_concurrency exceeds safe limit of "
                f"{MAX_PARALLEL_POLICY_CONCURRENCY}"
            ),
        )
    if policy.timeout_seconds <= 0:
        raise ContractViolationError(
            "PARALLEL_TIMEOUT_INVALID",
            "ParallelExecutionPolicy.timeout_seconds must be greater than zero",
        )
    if policy.cancellation_mode not in SUPPORTED_PARALLEL_CANCELLATION_MODES:
        supported = ", ".join(sorted(SUPPORTED_PARALLEL_CANCELLATION_MODES))
        raise ContractViolationError(
            "PARALLEL_CANCELLATION_MODE_INVALID",
            (
                "ParallelExecutionPolicy.cancellation_mode must be one of: "
                f"{supported}"
            ),
        )


def subagent_result_used_fallback(result: SubagentResult) -> bool:
    return any(
        error.startswith("JSON_TOO_LARGE") or error == "SUBAGENT_RESULT_SCHEMA_VIOLATION"
        for error in result.errors
    )


def tool_requests_manual_compact(result: ToolResult) -> bool:
    requested = result.metadata.get(ToolResultMeta.MANUAL_COMPACT_REQUESTED)
    return requested is True


# ---------------------------------------------------------------------------
# PolicyRunner — extracted from AgentOrchestrator._call_with_policies
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


class PolicyRunner:
    """Encapsulates timeout/retry logic for async operations.

    Extracted from ``AgentOrchestrator._call_with_policies`` so that both
    ``ToolDispatcher`` and ``ShellHandler`` can receive an object with clear
    semantics instead of a raw callback.
    """

    def __init__(
        self,
        *,
        timeout_policy: TimeoutPolicy | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._timeout_policy = timeout_policy
        self._retry_policy = retry_policy

    async def call(
        self,
        *,
        operation: str,
        report: ExecutionReport,
        latency_ms_sink: list[float] | None = None,
        retryable: bool = True,
        fn: Callable[[], Awaitable[_T]],
    ) -> _T:
        """Run *fn* with timeout/retry policies.

        This is the single authoritative implementation; all call-sites
        (orchestrator main loop, tool dispatch, shell handler) should use it.
        """
        attempt = 1
        while attempt <= MAX_RETRY_ATTEMPTS:
            timeout_seconds: float | None = None
            if self._timeout_policy is not None:
                timeout_seconds = self._timeout_policy.get_timeout(operation)
            attempt_started = time.monotonic()
            try:
                if timeout_seconds is not None and timeout_seconds > 0:
                    result = await asyncio.wait_for(fn(), timeout=timeout_seconds)
                else:
                    result = await fn()
                if latency_ms_sink is not None:
                    latency_ms_sink.append(
                        (time.monotonic() - attempt_started) * 1000
                    )
                return result
            except asyncio.CancelledError:
                if latency_ms_sink is not None:
                    latency_ms_sink.append(
                        (time.monotonic() - attempt_started) * 1000
                    )
                raise
            except asyncio.TimeoutError as exc:
                if latency_ms_sink is not None:
                    latency_ms_sink.append(
                        (time.monotonic() - attempt_started) * 1000
                    )
                timeout_exc = TimeoutError(f"{operation} timed out")
                if not self._should_retry(attempt, timeout_exc, retryable=retryable):
                    raise timeout_exc from exc
                report.add_warning(
                    f"retry:{operation}:attempt={attempt}:TimeoutError"
                )
                await self._sleep_before_retry(attempt)
                attempt += 1
            except Exception as exc:
                if latency_ms_sink is not None:
                    latency_ms_sink.append(
                        (time.monotonic() - attempt_started) * 1000
                    )
                if not self._should_retry(attempt, exc, retryable=retryable):
                    raise
                report.add_warning(
                    f"retry:{operation}:attempt={attempt}:{type(exc).__name__}"
                )
                await self._sleep_before_retry(attempt)
                attempt += 1
        raise RuntimeError(
            f"max retry attempts ({MAX_RETRY_ATTEMPTS}) exceeded for {operation}"
        )

    def _should_retry(
        self,
        attempt: int,
        error: Exception,
        *,
        retryable: bool,
    ) -> bool:
        if self._retry_policy is None or not retryable:
            return False
        return self._retry_policy.should_retry(attempt, error)

    async def _sleep_before_retry(self, attempt: int) -> None:
        if self._retry_policy is None:
            return
        delay = self._retry_policy.delay_seconds(attempt)
        if delay > 0:
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Pure utility functions extracted from AgentOrchestrator static methods
# ---------------------------------------------------------------------------


_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|secret|token|password|credential|auth[_-]?token"
    r"|bearer|private[_-]?key|access[_-]?key|client[_-]?secret)",
    re.IGNORECASE,
)


def redact_sensitive_keys(
    data: dict[str, Any] | list[Any],
    *,
    _depth: int = 0,
    _max_depth: int = 10,
) -> dict[str, Any] | list[Any]:
    """Return a copy of *data* with sensitive values replaced by ``[REDACTED]``.

    Accepts either a dict or a list at the root so JSON array tool payloads do not
    bypass redaction. Walks nested dicts/lists up to *_max_depth*.
    """
    if _depth > _max_depth:
        return {"__redacted__": "depth_limit"}
    if isinstance(data, list):
        return [
            redact_sensitive_keys(item, _depth=_depth + 1, _max_depth=_max_depth)
            if isinstance(item, (dict, list))
            else item
            for item in data
        ]
    result: dict[str, Any] = {}
    for key, value in data.items():
        if _SENSITIVE_KEY_RE.search(str(key)):
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = redact_sensitive_keys(
                value,
                _depth=_depth + 1,
                _max_depth=_max_depth,
            )
        elif isinstance(value, list):
            result[key] = [
                redact_sensitive_keys(item, _depth=_depth + 1, _max_depth=_max_depth)
                if isinstance(item, (dict, list))
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def string_preview(value: Any, *, limit: int = 600) -> str:
    """Truncate *value* to *limit* characters for log/event payloads."""
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def event_context_payload(
    context: AgentContext,
    *,
    phase: str,
    run_kind: RunKind | str | None = None,
    scope_id: str | None = None,
    iteration: int | None = None,
    correlation_id: str | None = None,
    attempt: int | None = None,
    source_component: str | None = None,
    node_id: str | None = None,
    workflow_id: str | None = None,
) -> dict[str, Any]:
    """Build the standard event payload envelope from an ``AgentContext``."""
    effective_run_kind = (
        run_kind.value
        if isinstance(run_kind, RunKind)
        else run_kind
        if isinstance(run_kind, str)
        else context.tool_context.metadata.get(ToolContextMeta.RUN_KIND)
    )
    payload: dict[str, Any] = {
        "agent_id": context.config.agent_id,
        "request_id": context.request_id,
        "trace_id": context.trace_id,
        "session_id": context.session_id,
        "parent_agent_id": context.parent_agent_id,
        "run_kind": effective_run_kind,
        "execution_mode": context.config.execution_mode.value,
        "phase": phase,
        "scope_id": scope_id,
        "iteration": iteration,
        "node_id": node_id,
        "workflow_id": workflow_id,
    }
    if correlation_id is not None:
        payload["correlation_id"] = correlation_id
    if attempt is not None:
        payload["attempt"] = attempt
    if source_component is not None:
        payload["source_component"] = source_component
    return payload


def tool_payload_summary(result: ToolResult) -> dict[str, Any]:
    """Summarise a ``ToolResult`` for event payloads.

    Sensitive keys are redacted from metadata and parsed JSON content.
    """
    summary: dict[str, Any] = {
        "tool_call_id": result.tool_call_id,
        "tool_name": result.tool_name,
        "is_error": result.is_error,
        "prompt_injection_signal": result.prompt_injection_signal,
        "latency_ms": result.latency_ms,
        "content_length": len(result.content),
        "content_preview": string_preview(result.content),
        "metadata": redact_sensitive_keys(dict(result.metadata)),
    }
    try:
        parsed = json.loads(result.content)
    except Exception:
        return summary
    summary["content_json"] = _summarize_json_payload(parsed)
    return summary


def _summarize_json_payload(parsed: Any, *, inline_limit: int = 1200) -> Any:
    if isinstance(parsed, (dict, list)):
        redacted = redact_sensitive_keys(parsed)
    else:
        redacted = parsed
    try:
        rendered = json.dumps(redacted, ensure_ascii=False, sort_keys=True)
    except Exception:
        return {"type": type(parsed).__name__}
    if len(rendered) <= inline_limit:
        return redacted
    if isinstance(redacted, dict):
        return {
            "type": "dict",
            "keys": list(redacted.keys())[:10],
            "key_count": len(redacted),
            "preview": string_preview(rendered, limit=inline_limit),
        }
    if isinstance(redacted, list):
        sample_types = [type(item).__name__ for item in redacted[:5]]
        return {
            "type": "list",
            "length": len(redacted),
            "sample_types": sample_types,
            "preview": string_preview(rendered, limit=inline_limit),
        }
    return string_preview(rendered, limit=inline_limit)


def resolve_shell_capability(
    config: AgentConfig,
    run_kind: RunKind,
) -> ShellToolConfig | None:
    """Return the shell capability config if shell is enabled for *run_kind*."""
    if not config.shell_tool_enabled_for_run(run_kind):
        return None
    return config.shell_tool_config


__all__ = [
    "PolicyRunner",
    "append_tool_results_as_messages",
    "build_llm_request_kwargs",
    "build_subagent_summary",
    "default_stop_reason_for_status",
    "ensure_terminal_report",
    "event_context_payload",
    "execution_status_from_subagent_status",
    "extract_task",
    "load_existing_plan",
    "merge_execution_report",
    "merge_nested_dict",
    "normalize_policy_decision",
    "redact_sensitive_keys",
    "parse_json_tool_call_candidates",
    "parse_legacy_tag_tool_call_candidates",
    "recover_tool_calls_from_assistant_text",
    "resolve_effective_llm_config",
    "resolve_max_tokens",
    "resolve_shell_capability",
    "serialize_messages_for_api",
    "session_refs",
    "string_preview",
    "subagent_result_used_fallback",
    "subagent_status_from_execution_status",
    "tool_payload_summary",
    "tool_requests_manual_compact",
    "validate_parallel_policy",
]
