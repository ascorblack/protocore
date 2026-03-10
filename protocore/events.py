"""EventBus for loose coupling between core components.

Events flow from orchestrator → subscribers (hooks, telemetry, logging).
No business logic; pure pub/sub.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoreEvent:
    """Base event emitted by the orchestrator loop."""

    name: str  # prefer EventName members; plain strings still accepted
    payload: dict[str, Any]
    ts: int = field(default_factory=lambda: int(time.time() * 1000))
    seq: int = 0
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class EventName(str, Enum):
    """Canonical event names.

    Using a ``StrEnum`` prevents silent typos — passing a raw string that
    doesn't match any member will be caught by linters and IDE inspections.
    Because ``EventName`` inherits from ``str``, members are fully
    interchangeable with plain strings at runtime (subscriptions, dict keys,
    comparisons, etc.).
    """

    def __str__(self) -> str:  # noqa: D105
        return self.value

    SESSION_START = "session.start"
    SESSION_END = "session.end"
    LOOP_ITERATION_START = "loop.iteration.start"
    LOOP_ITERATION_END = "loop.iteration.end"
    LOOP_BUDGET_EXCEEDED = "loop.budget.exceeded"
    LLM_CALL_START = "llm.call.start"
    LLM_CALL_END = "llm.call.end"
    LLM_CALL_FAILED = "llm.call.failed"
    LLM_REQUEST_PREPARED = "llm.request.prepared"
    LLM_STREAM_DELTA = "llm.stream.delta"
    LLM_STREAM_TOOL_CALL_DELTA = "llm.stream.tool_call.delta"
    LLM_STREAM_COMPLETED = "llm.stream.completed"
    LLM_OUTPUT_PARSED = "llm.output.parsed"
    LLM_OUTPUT_EMPTY = "llm.output.empty"
    TOOL_CALL_DETECTED = "tool.call.detected"
    TOOL_PREFLIGHT_START = "tool.preflight.start"
    TOOL_PREFLIGHT_END = "tool.preflight.end"
    TOOL_CALL_START = "tool.call.start"
    TOOL_CALL_END = "tool.call.end"
    TOOL_CALL_FAILED = "tool.call.failed"
    TOOL_DISPATCH_SELECTED = "tool.dispatch.selected"
    TOOL_VALIDATION_START = "tool.validation.start"
    TOOL_VALIDATION_END = "tool.validation.end"
    TOOL_APPROVAL_REQUIRED = "tool.approval.required"
    TOOL_APPROVAL_RESOLVED = "tool.approval.resolved"
    TOOL_EXECUTION_START = "tool.execution.start"
    TOOL_EXECUTION_END = "tool.execution.end"
    TOOL_RESULT_READY = "tool.result.ready"
    TOOL_RESULT_APPENDED = "tool.result.appended"
    TOOL_BUDGET_EXCEEDED = "tool.budget.exceeded"
    COMPRESSION_MICRO = "compression.micro"
    COMPRESSION_AUTO = "compression.auto"
    COMPRESSION_MANUAL = "compression.manual"
    COMPACTION_CHECK = "compaction.check"
    COMPACTION_AUTO_START = "compaction.auto.start"
    COMPACTION_MANUAL_START = "compaction.manual.start"
    COMPACTION_LLM_START = "compaction.llm.start"
    COMPACTION_LLM_DELTA = "compaction.llm.delta"
    COMPACTION_LLM_END = "compaction.llm.end"
    COMPACTION_SUMMARY_PARSE_START = "compaction.summary.parse.start"
    COMPACTION_SUMMARY_PARSE_END = "compaction.summary.parse.end"
    COMPACTION_SUMMARY_PARSE_FAILED = "compaction.summary.parse_failed"
    COMPACTION_END = "compaction.end"
    PLANNING_START = "planning.start"
    PLANNING_INPUT_PREPARED = "planning.input.prepared"
    PLANNING_END = "planning.end"
    PLANNING_FAILED = "planning.failed"
    WORKFLOW_START = "workflow.start"
    WORKFLOW_END = "workflow.end"
    SUBAGENT_SELECTION_START = "subagent.selection.start"
    SUBAGENT_SELECTION_END = "subagent.selection.end"
    SUBAGENT_DISPATCH_START = "subagent.dispatch.start"
    SUBAGENT_START = "subagent.start"
    SUBAGENT_RESULT_RAW = "subagent.result.raw"
    SUBAGENT_RESULT_PARSE_START = "subagent.result.parse.start"
    SUBAGENT_RESULT_PARSE_END = "subagent.result.parse.end"
    SUBAGENT_RESULT_FALLBACK = "subagent.result.fallback"
    SUBAGENT_END = "subagent.end"
    PARALLEL_RUN_START = "parallel.run.start"
    PARALLEL_SLOT_ACQUIRED = "parallel.slot.acquired"
    PARALLEL_SLOT_RELEASED = "parallel.slot.released"
    PARALLEL_CHILD_TIMEOUT = "parallel.child.timeout"
    PARALLEL_CHILD_CANCELLED = "parallel.child.cancelled"
    PARALLEL_CHILD_FAILED = "parallel.child.failed"
    PARALLEL_RUN_END = "parallel.run.end"
    SYNTHESIS_START = "synthesis.start"
    SYNTHESIS_INPUT_COLLECTED = "synthesis.input.collected"
    SYNTHESIS_MERGE_START = "synthesis.merge.start"
    SYNTHESIS_MERGE_END = "synthesis.merge.end"
    SYNTHESIS_END = "synthesis.end"
    DESTRUCTIVE_ACTION = "safety.destructive_action"
    INJECTION_SIGNAL = "safety.injection_signal"
    CANCELLATION = "loop.cancelled"
    ERROR = "loop.error"
    SKILL_INDEX_INJECTED = "skill.index.injected"
    SKILL_LOAD_START = "skill.load.start"
    SKILL_LOAD_END = "skill.load.end"
    SKILL_BUDGET_EXCEEDED = "skill.budget.exceeded"


# Backward-compatible aliases so existing ``from .events import EV_*``
# imports continue to work without changes.
EV_SESSION_START = EventName.SESSION_START
EV_SESSION_END = EventName.SESSION_END
EV_LOOP_ITERATION_START = EventName.LOOP_ITERATION_START
EV_LOOP_ITERATION_END = EventName.LOOP_ITERATION_END
EV_LOOP_BUDGET_EXCEEDED = EventName.LOOP_BUDGET_EXCEEDED
EV_LOOP_ITERATION = EventName.LOOP_ITERATION_END
EV_LLM_CALL_START = EventName.LLM_CALL_START
EV_LLM_CALL_END = EventName.LLM_CALL_END
EV_LLM_CALL_FAILED = EventName.LLM_CALL_FAILED
EV_LLM_REQUEST_PREPARED = EventName.LLM_REQUEST_PREPARED
EV_LLM_STREAM_DELTA = EventName.LLM_STREAM_DELTA
EV_LLM_STREAM_TOOL_CALL_DELTA = EventName.LLM_STREAM_TOOL_CALL_DELTA
EV_LLM_STREAM_COMPLETED = EventName.LLM_STREAM_COMPLETED
EV_LLM_OUTPUT_PARSED = EventName.LLM_OUTPUT_PARSED
EV_LLM_OUTPUT_EMPTY = EventName.LLM_OUTPUT_EMPTY
EV_TOOL_CALL_DETECTED = EventName.TOOL_CALL_DETECTED
EV_TOOL_PREFLIGHT_START = EventName.TOOL_PREFLIGHT_START
EV_TOOL_PREFLIGHT_END = EventName.TOOL_PREFLIGHT_END
EV_TOOL_CALL_START = EventName.TOOL_CALL_START
EV_TOOL_CALL_END = EventName.TOOL_CALL_END
EV_TOOL_CALL_FAILED = EventName.TOOL_CALL_FAILED
EV_TOOL_DISPATCH_SELECTED = EventName.TOOL_DISPATCH_SELECTED
EV_TOOL_VALIDATION_START = EventName.TOOL_VALIDATION_START
EV_TOOL_VALIDATION_END = EventName.TOOL_VALIDATION_END
EV_TOOL_APPROVAL_REQUIRED = EventName.TOOL_APPROVAL_REQUIRED
EV_TOOL_APPROVAL_RESOLVED = EventName.TOOL_APPROVAL_RESOLVED
EV_TOOL_EXECUTION_START = EventName.TOOL_EXECUTION_START
EV_TOOL_EXECUTION_END = EventName.TOOL_EXECUTION_END
EV_TOOL_RESULT_READY = EventName.TOOL_RESULT_READY
EV_TOOL_RESULT_APPENDED = EventName.TOOL_RESULT_APPENDED
EV_TOOL_BUDGET_EXCEEDED = EventName.TOOL_BUDGET_EXCEEDED
EV_COMPRESSION_MICRO = EventName.COMPRESSION_MICRO
EV_COMPRESSION_AUTO = EventName.COMPRESSION_AUTO
EV_COMPRESSION_MANUAL = EventName.COMPRESSION_MANUAL
EV_COMPACTION_CHECK = EventName.COMPACTION_CHECK
EV_COMPACTION_AUTO_START = EventName.COMPACTION_AUTO_START
EV_COMPACTION_MANUAL_START = EventName.COMPACTION_MANUAL_START
EV_COMPACTION_LLM_START = EventName.COMPACTION_LLM_START
EV_COMPACTION_LLM_DELTA = EventName.COMPACTION_LLM_DELTA
EV_COMPACTION_LLM_END = EventName.COMPACTION_LLM_END
EV_COMPACTION_SUMMARY_PARSE_START = EventName.COMPACTION_SUMMARY_PARSE_START
EV_COMPACTION_SUMMARY_PARSE_END = EventName.COMPACTION_SUMMARY_PARSE_END
EV_COMPACTION_SUMMARY_PARSE_FAILED = EventName.COMPACTION_SUMMARY_PARSE_FAILED
EV_COMPACTION_END = EventName.COMPACTION_END
EV_PLANNING_START = EventName.PLANNING_START
EV_PLANNING_INPUT_PREPARED = EventName.PLANNING_INPUT_PREPARED
EV_PLANNING_END = EventName.PLANNING_END
EV_PLANNING_FAILED = EventName.PLANNING_FAILED
EV_WORKFLOW_START = EventName.WORKFLOW_START
EV_WORKFLOW_END = EventName.WORKFLOW_END
EV_SUBAGENT_SELECTION_START = EventName.SUBAGENT_SELECTION_START
EV_SUBAGENT_SELECTION_END = EventName.SUBAGENT_SELECTION_END
EV_SUBAGENT_DISPATCH_START = EventName.SUBAGENT_DISPATCH_START
EV_SUBAGENT_START = EventName.SUBAGENT_START
EV_SUBAGENT_RESULT_RAW = EventName.SUBAGENT_RESULT_RAW
EV_SUBAGENT_RESULT_PARSE_START = EventName.SUBAGENT_RESULT_PARSE_START
EV_SUBAGENT_RESULT_PARSE_END = EventName.SUBAGENT_RESULT_PARSE_END
EV_SUBAGENT_RESULT_FALLBACK = EventName.SUBAGENT_RESULT_FALLBACK
EV_SUBAGENT_END = EventName.SUBAGENT_END
EV_PARALLEL_RUN_START = EventName.PARALLEL_RUN_START
EV_PARALLEL_SLOT_ACQUIRED = EventName.PARALLEL_SLOT_ACQUIRED
EV_PARALLEL_SLOT_RELEASED = EventName.PARALLEL_SLOT_RELEASED
EV_PARALLEL_CHILD_TIMEOUT = EventName.PARALLEL_CHILD_TIMEOUT
EV_PARALLEL_CHILD_CANCELLED = EventName.PARALLEL_CHILD_CANCELLED
EV_PARALLEL_CHILD_FAILED = EventName.PARALLEL_CHILD_FAILED
EV_PARALLEL_RUN_END = EventName.PARALLEL_RUN_END
EV_SYNTHESIS_START = EventName.SYNTHESIS_START
EV_SYNTHESIS_INPUT_COLLECTED = EventName.SYNTHESIS_INPUT_COLLECTED
EV_SYNTHESIS_MERGE_START = EventName.SYNTHESIS_MERGE_START
EV_SYNTHESIS_MERGE_END = EventName.SYNTHESIS_MERGE_END
EV_SYNTHESIS_END = EventName.SYNTHESIS_END
EV_DESTRUCTIVE_ACTION = EventName.DESTRUCTIVE_ACTION
EV_INJECTION_SIGNAL = EventName.INJECTION_SIGNAL
EV_CANCELLATION = EventName.CANCELLATION
EV_ERROR = EventName.ERROR
EV_SKILL_INDEX_INJECTED = EventName.SKILL_INDEX_INJECTED
EV_SKILL_LOAD_START = EventName.SKILL_LOAD_START
EV_SKILL_LOAD_END = EventName.SKILL_LOAD_END
EV_SKILL_BUDGET_EXCEEDED = EventName.SKILL_BUDGET_EXCEEDED

# Set of all known event string values for runtime validation.
_KNOWN_EVENT_NAMES: frozenset[str] = frozenset(m.value for m in EventName)
MAX_HANDLERS_PER_EVENT = 100
MAX_EVENT_PAYLOAD_SIZE = 10_000

# ---------------------------------------------------------------------------
# Handler type alias
# ---------------------------------------------------------------------------

EventHandler = Callable[[CoreEvent], Awaitable[None]]
ErrorSink = Callable[[CoreEvent, Exception], Awaitable[None] | None]
EventFilter = Callable[[CoreEvent], bool]


def _truncate_payload_value(value: Any, *, limit: int = MAX_EVENT_PAYLOAD_SIZE) -> Any:
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        omitted = len(value) - limit
        return value[:limit] + f"[truncated, {omitted} chars omitted]"
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        rendered = str(value)
    if len(rendered) <= limit:
        return value
    omitted = len(rendered) - limit
    return rendered[:limit] + f"[truncated, {omitted} chars omitted]"


@dataclass(frozen=True, slots=True)
class SubscriptionToken:
    event_name: str
    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True, slots=True)
class ErrorSinkToken:
    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Async pub/sub event bus.

    Handlers are registered per event name or with wildcard ``"*"`` to
    receive all events. Errors in handlers are logged and swallowed so they
    never break the immutable loop.
    """

    def __init__(self, *, parent_bus: EventBus | None = None) -> None:
        self._handlers: dict[str, list[tuple[SubscriptionToken, EventHandler, EventFilter | None]]] = defaultdict(list)
        self._error_sinks: list[tuple[ErrorSinkToken, ErrorSink, EventFilter | None]] = []
        self._parent_bus = parent_bus
        self._seq = 0

    def subscribe(
        self,
        event_name: str,
        handler: EventHandler,
        *,
        event_filter: EventFilter | None = None,
    ) -> SubscriptionToken:
        """Register *handler* for *event_name* or ``"*"`` for all events.

        Raises ``RuntimeError`` when the per-event handler cap is exceeded so
        callers do not silently proceed with an inactive subscription token.
        """
        handlers = self._handlers[event_name]
        if any(existing is handler for _, existing, _ in handlers):
            logger.debug(
                "EventBus ignored duplicate subscription: event=%s handler=%r",
                event_name,
                handler,
            )
            return next(token for token, existing, _ in handlers if existing is handler)
        if len(handlers) >= MAX_HANDLERS_PER_EVENT:
            logger.error(
                "EventBus handler limit exceeded: event=%s limit=%d",
                event_name,
                MAX_HANDLERS_PER_EVENT,
            )
            raise RuntimeError(
                f"EventBus handler limit exceeded for event={event_name!r} "
                f"(limit={MAX_HANDLERS_PER_EVENT}). "
                "Consider increasing MAX_HANDLERS_PER_EVENT or auditing subscriptions."
            )
        token = SubscriptionToken(event_name=event_name)
        handlers.append((token, handler, event_filter))
        return token

    def unsubscribe(
        self,
        event_name: str,
        handler_or_token: EventHandler | SubscriptionToken,
    ) -> None:
        """Remove a previously registered handler."""
        handlers = self._handlers.get(event_name)
        if not handlers:
            return
        target = next(
            (
                entry
                for entry in handlers
                if entry[0] is handler_or_token or entry[1] is handler_or_token
            ),
            None,
        )
        if target is None:
            return
        handlers.remove(target)
        if not handlers:
            self._handlers.pop(event_name, None)

    def cleanup_stale_handlers(self) -> int:
        """Remove empty handler buckets left behind by unsubscribe operations."""
        removed = 0
        for event_name in list(self._handlers):
            if self._handlers[event_name]:
                continue
            self._handlers.pop(event_name, None)
            removed += 1
        return removed

    def handler_count(self, event_name: str = "*") -> int:
        """Return the number of handlers registered for *event_name*.

        Useful for load balancing: the service can pick the orchestrator
        (or event bus) with the fewest active subscriptions. For the default
        orchestrator setup, pass ``"*"`` (wildcard telemetry subscriptions).
        """
        return len(self._handlers.get(event_name, []))

    async def emit(self, event: CoreEvent) -> None:
        """Emit *event* to all matching and wildcard handlers."""
        targets = list(self._handlers.get(event.name, []))
        targets.extend(self._handlers.get("*", []))
        seen_handlers: set[int] = set()
        for _token, handler, event_filter in targets:
            handler_id = id(handler)
            if handler_id in seen_handlers:
                continue
            if event_filter is not None and not event_filter(event):
                continue
            seen_handlers.add(handler_id)
            await self._safe_call(handler, event)
        if self._parent_bus is not None:
            await self._parent_bus.emit(event)

    async def emit_simple(self, name: str, **payload: Any) -> None:
        """Convenience: emit with keyword-arg payload.

        Logs a warning if *name* is not a known ``EventName`` member, which
        helps catch typos early without breaking at runtime.
        """
        if name not in _KNOWN_EVENT_NAMES and name != "*":
            logger.warning("emit_simple called with unknown event name: %r", name)
        truncated_payload = {
            key: _truncate_payload_value(value)
            for key, value in payload.items()
        }
        self._seq += 1
        run_id = payload.get("request_id")
        await self.emit(
            CoreEvent(
                name=name,
                seq=self._seq,
                run_id=str(run_id) if run_id is not None else str(uuid.uuid4()),
                payload=truncated_payload,
            )
        )

    def set_error_sink(self, sink: ErrorSink | None) -> None:
        """Register a sink that observes handler failures."""
        self._error_sinks = []
        if sink is not None:
            self.push_error_sink(sink)

    def push_error_sink(
        self,
        sink: ErrorSink,
        *,
        event_filter: EventFilter | None = None,
    ) -> ErrorSinkToken:
        token = ErrorSinkToken()
        self._error_sinks.append((token, sink, event_filter))
        return token

    def pop_error_sink(self, token: ErrorSinkToken) -> None:
        self._error_sinks = [
            entry for entry in self._error_sinks if entry[0] is not token
        ]

    async def _safe_call(self, handler: EventHandler, event: CoreEvent) -> None:
        try:
            await handler(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("EventBus handler error for event=%s", event.name)
            if self._error_sinks:
                try:
                    for _, sink, event_filter in list(self._error_sinks):
                        if event_filter is not None and not event_filter(event):
                            continue
                        maybe_awaitable = sink(event, exc)
                        if maybe_awaitable is not None:
                            await maybe_awaitable
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("EventBus error sink failed for event=%s", event.name)


# ---------------------------------------------------------------------------
# Registry helper (explicitly constructed via DI)
# ---------------------------------------------------------------------------


@dataclass
class BusRegistry:
    """Holds named event bus instances for DI scenarios."""

    _buses: dict[str, EventBus] = field(default_factory=dict)

    def get_or_create(self, name: str = "default") -> EventBus:
        if name not in self._buses:
            self._buses[name] = EventBus()
        return self._buses[name]


def create_event_bus() -> EventBus:
    """Create and return a fresh EventBus instance."""
    return EventBus()


def get_event_bus(name: str = "default") -> EventBus:
    """Deprecated alias for ``create_event_bus``.

    The *name* parameter is ignored and kept only for API compatibility.
    """
    _ = name
    warnings.warn(
        "get_event_bus(name=...) is deprecated; use create_event_bus() or BusRegistry.get_or_create().",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_event_bus()
