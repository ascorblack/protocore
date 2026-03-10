"""Helpers for attaching stable context to log records.

The core library uses plain ``logging``. These helpers keep request/session
identifiers in ``extra`` so services with structured logging handlers can
correlate orchestration, LLM, and shell events without changing message text.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import AgentContext, ExecutionReport


def merge_log_context(
    logging_context: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Merge an optional logging context with additional structured fields."""
    merged: dict[str, Any] = {}
    if logging_context:
        for key, value in logging_context.items():
            if value is not None:
                merged[str(key)] = value
    for key, value in extra.items():
        if value is not None:
            merged[key] = value
    return merged


def context_log_extra(
    context: AgentContext,
    *,
    report: ExecutionReport | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build standard ``logging.extra`` fields from an ``AgentContext``."""
    payload = {
        "agent_id": context.config.agent_id,
        "request_id": context.request_id,
        "trace_id": context.trace_id,
        "session_id": context.session_id,
        "parent_agent_id": context.parent_agent_id,
        "execution_mode": context.config.execution_mode.value,
    }
    if report is not None:
        payload["run_kind"] = report.run_kind.value
        payload["report_status"] = report.status.value
    return merge_log_context(payload, **extra)
