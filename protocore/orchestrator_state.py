"""Helpers extracted from ``orchestrator.py`` for runtime state handling."""
from __future__ import annotations

import copy
import uuid
from datetime import datetime, timezone
from typing import Any

from .types import (
    AgentConfig,
    AgentContext,
    AgentContextMeta,
    ExecutionReport,
    Message,
    MessageList,
    SessionSnapshot,
    ToolContextMeta,
)

_LEGACY_ACTIVE_CHILD_AGENT_IDS_KEY = "_active_child_agent_ids"


def ensure_active_child_agent_ids(metadata: dict[str, Any]) -> list[str]:
    """Normalize legacy/new active-child metadata keys and return the list."""
    current = metadata.get(AgentContextMeta.ACTIVE_CHILD_AGENT_IDS.value)
    if isinstance(current, list):
        metadata[_LEGACY_ACTIVE_CHILD_AGENT_IDS_KEY] = list(current)
        return list(current)

    legacy = metadata.pop(_LEGACY_ACTIVE_CHILD_AGENT_IDS_KEY, None)
    if isinstance(legacy, list):
        metadata[AgentContextMeta.ACTIVE_CHILD_AGENT_IDS.value] = list(legacy)
        metadata[_LEGACY_ACTIVE_CHILD_AGENT_IDS_KEY] = list(legacy)
        return list(legacy)

    metadata[AgentContextMeta.ACTIVE_CHILD_AGENT_IDS.value] = []
    metadata[_LEGACY_ACTIVE_CHILD_AGENT_IDS_KEY] = []
    return []


def get_active_child_agent_ids(metadata: dict[str, Any]) -> list[str]:
    """Return active child ids while transparently migrating the legacy key."""
    value = ensure_active_child_agent_ids(metadata)
    return list(value)


def set_active_child_agent_ids(metadata: dict[str, Any], child_ids: list[str]) -> None:
    """Store active child ids under canonical and deprecated compatibility keys."""
    normalized = list(child_ids)
    metadata[AgentContextMeta.ACTIVE_CHILD_AGENT_IDS.value] = normalized
    metadata[_LEGACY_ACTIVE_CHILD_AGENT_IDS_KEY] = list(normalized)


def set_queue_wait_metric(*, report: ExecutionReport, context: AgentContext) -> None:
    """Populate ``report.queue_wait_ms`` from agent metadata when present."""
    queue_wait = context.metadata.get(AgentContextMeta.QUEUE_WAIT_MS)
    if queue_wait is None:
        return
    try:
        report.queue_wait_ms = float(queue_wait)
    except (TypeError, ValueError):
        report.warnings.append("queue_wait_ms_invalid")


def accumulate_llm_usage(
    *,
    report: ExecutionReport,
    context: AgentContext,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    reasoning_tokens: int = 0,
) -> None:
    """Accumulate token/cost accounting for one LLM turn."""
    report.input_tokens += input_tokens
    report.output_tokens += output_tokens
    report.cached_tokens += cached_tokens
    report.reasoning_tokens += reasoning_tokens
    if context.config.cost_per_token is None:
        return

    non_cached_input = max(0, input_tokens - cached_tokens)
    cost = (non_cached_input + output_tokens) * context.config.cost_per_token
    if cached_tokens > 0 and context.config.cost_per_cached_token is not None:
        cost += cached_tokens * context.config.cost_per_cached_token
    report.estimated_cost = (report.estimated_cost or 0.0) + cost


def build_subagent_context(
    *,
    parent_context: AgentContext,
    child_config: AgentConfig,
    task_text: str,
) -> AgentContext:
    """Create an isolated child context while preserving parent session links."""
    child_request_id = str(uuid.uuid4())
    child_execution_metadata_ref = f"request:{child_request_id}:metadata"
    child_tool_metadata = copy.deepcopy(parent_context.tool_context.metadata)
    child_tool_metadata.update(
        {
            ToolContextMeta.REQUEST_ID: child_request_id,
            ToolContextMeta.EXECUTION_METADATA_REF: child_execution_metadata_ref,
            ToolContextMeta.PARENT_AGENT_ID: parent_context.config.agent_id,
        }
    )
    inherited_decisions = parent_context.metadata.get(
        AgentContextMeta.SHELL_APPROVAL_DECISIONS
    )
    inherited_pending_map = parent_context.metadata.get("subagent_pending_shell_approvals")
    child_pending = None
    if isinstance(inherited_pending_map, dict):
        pending_for_child = inherited_pending_map.get(child_config.agent_id)
        if isinstance(pending_for_child, dict):
            child_pending = pending_for_child

    child_metadata: dict[str, Any] = {
        "parent_execution_mode": parent_context.config.execution_mode.value,
        "bypass_explicit": True,
        "bypass_reason": "subagent_dispatch",
        "parent_request_id": parent_context.request_id,
    }
    if isinstance(inherited_decisions, dict):
        child_metadata[AgentContextMeta.SHELL_APPROVAL_DECISIONS] = copy.deepcopy(
            inherited_decisions
        )
    if child_pending is not None:
        child_metadata[AgentContextMeta.PENDING_SHELL_APPROVAL] = copy.deepcopy(
            child_pending
        )

    return AgentContext(
        session_id=parent_context.session_id,
        trace_id=parent_context.trace_id,
        request_id=child_request_id,
        parent_agent_id=parent_context.config.agent_id,
        message_history_ref=parent_context.message_history_ref,
        execution_metadata_ref=child_execution_metadata_ref,
        config=child_config,
        tool_context=parent_context.tool_context.model_copy(
            deep=True,
            update={
                "agent_id": child_config.agent_id,
                "metadata": child_tool_metadata,
            },
        ),
        messages=MessageList([Message(role="user", content=task_text)]),
        metadata=child_metadata,
    )


def build_session_snapshot(
    *,
    context: AgentContext,
    report: ExecutionReport,
    message_history_ref: str,
    execution_metadata_ref: str,
    existing_snapshot: SessionSnapshot | None,
) -> SessionSnapshot:
    """Build the persisted session snapshot while preserving ``created_at``."""
    return SessionSnapshot(
        session_id=context.session_id,
        trace_id=context.trace_id,
        agent_id=context.config.agent_id,
        message_history_ref=message_history_ref,
        execution_metadata_ref=execution_metadata_ref,
        messages=list(context.messages),
        execution_report_id=context.request_id,
        created_at=(
            existing_snapshot.created_at
            if existing_snapshot is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        metadata={
            "status": report.status.value,
            "stop_reason": report.stop_reason.value if report.stop_reason else None,
            "warnings": list(report.warnings),
            AgentContextMeta.PENDING_SHELL_APPROVAL: context.metadata.get(
                AgentContextMeta.PENDING_SHELL_APPROVAL
            ),
        },
    )
