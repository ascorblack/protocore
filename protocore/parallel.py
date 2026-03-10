"""Parallel orchestration helpers extracted from `orchestrator.py`."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .context import CancellationContext
from .events import (
    EV_PARALLEL_CHILD_CANCELLED,
    EV_PARALLEL_CHILD_FAILED,
    EV_PARALLEL_CHILD_TIMEOUT,
    EV_PARALLEL_SLOT_ACQUIRED,
    EV_PARALLEL_SLOT_RELEASED,
    EV_SYNTHESIS_END,
    EV_SYNTHESIS_INPUT_COLLECTED,
    EV_SYNTHESIS_MERGE_END,
    EV_SYNTHESIS_MERGE_START,
    EV_SYNTHESIS_START,
    EventBus,
)
from .factories import make_control_envelope
from .orchestrator_errors import ContractViolationError
from .orchestrator_utils import (
    build_subagent_summary,
    execution_status_from_subagent_status,
    subagent_status_from_execution_status,
    validate_parallel_policy,
)
from .protocols import ParallelExecutionPolicy, Transport
from .types import (
    AgentContext,
    ControlCommand,
    ExecutionMode,
    ExecutionReport,
    ExecutionStatus,
    Result,
    RunKind,
    SubagentResult,
    SubagentRunSummary,
    SubagentStatus,
)

if TYPE_CHECKING:
    from .orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SubagentRunOutcome:
    agent_id: str
    structured_result: SubagentResult | None
    execution_status: ExecutionStatus
    error_message: str | None
    started_at: str
    finished_at: str
    duration_ms: float
    child_report: ExecutionReport | None
    pending_shell_approval: dict[str, Any] | None = None


class ParallelSubagentRunner:
    """Run multiple subagent orchestrators concurrently."""

    def __init__(
        self,
        *,
        policy: ParallelExecutionPolicy,
        orchestrator_factory: Any,
        transport: Transport | None = None,
        parent_context: AgentContext | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._policy = policy
        self._factory = orchestrator_factory
        self._transport = transport
        self._parent_context = parent_context
        self._bus = event_bus

    async def _emit(self, name: str, **payload: Any) -> None:
        if self._bus is None or self._parent_context is None:
            return
        base = {
            "agent_id": self._parent_context.config.agent_id,
            "request_id": self._parent_context.request_id,
            "trace_id": self._parent_context.trace_id,
            "session_id": self._parent_context.session_id,
            "parent_agent_id": self._parent_context.parent_agent_id,
            "run_kind": RunKind.LEADER.value,
            "execution_mode": self._parent_context.config.execution_mode.value,
            "phase": "synthesis",
            "scope_id": f"parallel:{self._parent_context.request_id}",
            "iteration": None,
            "node_id": None,
            "workflow_id": None,
            "source_component": "parallel",
        }
        await self._bus.emit_simple(name, **base, **payload)

    async def run_parallel(
        self,
        tasks: list[tuple[str, AgentContext]],
        cancel_ctx: CancellationContext | None = None,
        report: ExecutionReport | None = None,
    ) -> tuple[SubagentResult, list[SubagentRunSummary]]:
        """Run tasks in parallel, respect max_concurrency and timeout.

        Cancellation is treated as a control-flow signal and must not be
        normalized into a failed subagent result.
        """
        validate_parallel_policy(self._policy)
        ordered_tasks = sorted(tasks, key=lambda item: item[0])
        ordered_agent_ids = [agent_id for agent_id, _ in ordered_tasks]
        if len(ordered_agent_ids) != len(set(ordered_agent_ids)):
            raise ContractViolationError(
                "PARALLEL_AGENT_IDS_DUPLICATE",
                "parallel tasks must not contain duplicate agent_ids",
            )
        if report is not None:
            report.subagents_parallel_max = max(
                report.subagents_parallel_max, len(ordered_tasks)
            )
        semaphore = asyncio.Semaphore(self._policy.max_concurrency)
        timeout = self._policy.timeout_seconds
        cancellation_mode = self._policy.cancellation_mode
        active_child_ids = set(ordered_agent_ids)
        cancel_lock = asyncio.Lock()
        cancel_sent_to: set[str] = set()
        local_tasks: dict[str, asyncio.Task[SubagentRunOutcome]] = {}

        async def send_cancel_to_active_children() -> None:
            if self._transport is None or self._parent_context is None:
                return
            async with cancel_lock:
                pending = sorted(active_child_ids - cancel_sent_to)
                cancel_sent_to.update(pending)
            for agent_id in pending:
                envelope = make_control_envelope(
                    sender_id=self._parent_context.config.agent_id,
                    recipient_id=agent_id,
                    command=ControlCommand.CANCEL,
                    trace_id=self._parent_context.trace_id,
                    session_id=self._parent_context.session_id,
                    report=report,
                )
                try:
                    await self._transport.send(envelope, destination=agent_id)
                except Exception as exc:
                    logger.exception(
                        "Failed to propagate parallel cancel to child=%s", agent_id
                    )
                    if report is not None:
                        report.add_warning(
                            f"parallel_cancel_propagation_failed:{agent_id}:{type(exc).__name__}"
                        )

        async def watch_parent_cancellation() -> None:
            if cancel_ctx is None:
                return
            await cancel_ctx.wait()
            await send_cancel_to_active_children()
            async with cancel_lock:
                pending_local_tasks = [
                    task for task in local_tasks.values() if not task.done()
                ]
            for task in pending_local_tasks:
                task.cancel()

        async def run_one(
            agent_id: str, ctx: AgentContext
        ) -> SubagentRunOutcome:
            async with semaphore:
                await self._emit(
                    EV_PARALLEL_SLOT_ACQUIRED,
                    child_agent_id=agent_id,
                    slot_index=max(0, self._policy.max_concurrency - semaphore._value - 1),
                )
                orch: AgentOrchestrator = self._factory()
                started_wall = datetime.now(timezone.utc).isoformat()
                started = time.monotonic()
                try:
                    result, rep = await asyncio.wait_for(
                        orch.run(ctx, run_kind=RunKind.SUBAGENT, cancel_ctx=cancel_ctx),
                        timeout=timeout,
                    )
                    pending_shell_approval = (
                        result.metadata.get("pending_shell_approval")
                        if isinstance(result.metadata, dict)
                        else None
                    )
                    parent_cancelled = cancel_ctx is not None and cancel_ctx.is_cancelled
                    effective_status = rep.status
                    if (
                        effective_status == ExecutionStatus.CANCELLED
                        and not parent_cancelled
                        and (time.monotonic() - started) >= timeout
                    ):
                        effective_status = ExecutionStatus.TIMEOUT
                    if (
                        effective_status == ExecutionStatus.CANCELLED
                        and (parent_cancelled or cancellation_mode == "propagate")
                    ):
                        await send_cancel_to_active_children()
                    structured_result = SubagentResult.parse_with_fallback(
                        result.content,
                        agent_id=agent_id,
                    )
                    if effective_status in (
                        ExecutionStatus.CANCELLED,
                        ExecutionStatus.FAILED,
                        ExecutionStatus.TIMEOUT,
                    ):
                        logger.warning(
                            "Parallel subagent %s: status=%s",
                            agent_id,
                            effective_status,
                        )
                        subagent_status = subagent_status_from_execution_status(
                            effective_status
                        )
                        finished_wall = (
                            rep.finished_at or datetime.now(timezone.utc).isoformat()
                        )
                        return SubagentRunOutcome(
                            agent_id=agent_id,
                            structured_result=structured_result.model_copy(
                                update={
                                    "status": subagent_status,
                                    "errors": structured_result.errors
                                    or [rep.error_message or effective_status.value],
                                }
                            ),
                            execution_status=effective_status,
                            error_message=rep.error_message,
                            started_at=started_wall,
                            finished_at=finished_wall,
                            duration_ms=rep.duration_ms or (time.monotonic() - started) * 1000,
                            child_report=rep,
                            pending_shell_approval=(
                                pending_shell_approval
                                if isinstance(pending_shell_approval, dict)
                                else None
                            ),
                        )
                    finished_wall = rep.finished_at or datetime.now(timezone.utc).isoformat()
                    return SubagentRunOutcome(
                        agent_id=agent_id,
                        structured_result=structured_result.model_copy(
                            update={
                                "artifacts": structured_result.artifacts or result.artifacts,
                            }
                        ),
                        execution_status=effective_status,
                        error_message=rep.error_message,
                        started_at=started_wall,
                        finished_at=finished_wall,
                        duration_ms=rep.duration_ms or (time.monotonic() - started) * 1000,
                        child_report=rep,
                        pending_shell_approval=(
                            pending_shell_approval
                            if isinstance(pending_shell_approval, dict)
                            else None
                        ),
                    )
                except asyncio.TimeoutError:
                    logger.warning("Parallel subagent timeout: agent=%s", agent_id)
                    await self._emit(
                        EV_PARALLEL_CHILD_TIMEOUT,
                        child_agent_id=agent_id,
                        timeout_seconds=timeout,
                    )
                    finished_wall = datetime.now(timezone.utc).isoformat()
                    return SubagentRunOutcome(
                        agent_id=agent_id,
                        structured_result=None,
                        execution_status=ExecutionStatus.TIMEOUT,
                        error_message="timeout",
                        started_at=started_wall,
                        finished_at=finished_wall,
                        duration_ms=(time.monotonic() - started) * 1000,
                        child_report=None,
                    )
                except asyncio.CancelledError:
                    await self._emit(
                        EV_PARALLEL_CHILD_CANCELLED,
                        child_agent_id=agent_id,
                        reason=cancel_ctx.reason if cancel_ctx is not None else "cancelled",
                    )
                    if (
                        (cancel_ctx is not None and cancel_ctx.is_cancelled)
                        or cancellation_mode == "propagate"
                    ):
                        await send_cancel_to_active_children()
                    finished_wall = datetime.now(timezone.utc).isoformat()
                    return SubagentRunOutcome(
                        agent_id=agent_id,
                        structured_result=None,
                        execution_status=ExecutionStatus.CANCELLED,
                        error_message="cancelled",
                        started_at=started_wall,
                        finished_at=finished_wall,
                        duration_ms=(time.monotonic() - started) * 1000,
                        child_report=None,
                    )
                except Exception:
                    logger.exception("Parallel subagent error: agent=%s", agent_id)
                    await self._emit(
                        EV_PARALLEL_CHILD_FAILED,
                        child_agent_id=agent_id,
                        error_message=f"agent {agent_id} failed",
                    )
                    finished_wall = datetime.now(timezone.utc).isoformat()
                    return SubagentRunOutcome(
                        agent_id=agent_id,
                        structured_result=SubagentResult(
                            status=SubagentStatus.FAILED,
                            summary="[error]",
                            errors=[f"agent {agent_id} failed"],
                        ),
                        execution_status=ExecutionStatus.FAILED,
                        error_message=f"agent {agent_id} failed",
                        started_at=started_wall,
                        finished_at=finished_wall,
                        duration_ms=(time.monotonic() - started) * 1000,
                        child_report=None,
                    )
                finally:
                    await self._emit(
                        EV_PARALLEL_SLOT_RELEASED,
                        child_agent_id=agent_id,
                        duration_ms=(time.monotonic() - started) * 1000,
                    )
                    async with cancel_lock:
                        active_child_ids.discard(agent_id)

        task_list: list[asyncio.Task[SubagentRunOutcome]] = []
        for agent_id, ctx in ordered_tasks:
            task = asyncio.create_task(run_one(agent_id, ctx))
            local_tasks[agent_id] = task
            task_list.append(task)
        cancel_monitor: asyncio.Task[None] | None = None
        if cancel_ctx is not None:
            cancel_monitor = asyncio.create_task(watch_parent_cancellation())
        try:
            raw_results = await asyncio.gather(*task_list, return_exceptions=True)
        finally:
            if cancel_monitor is not None:
                cancel_monitor.cancel()
                await asyncio.gather(cancel_monitor, return_exceptions=True)
        normalized_results: list[SubagentRunOutcome] = []
        for index, outcome in enumerate(raw_results):
            if isinstance(outcome, SubagentRunOutcome):
                normalized_results.append(outcome)
                continue
            agent_id = ordered_tasks[index][0]
            if isinstance(outcome, asyncio.CancelledError):
                # Preserve cancellation semantics for parallel runs.
                raise outcome
            exc_str = str(outcome) if isinstance(outcome, Exception) else repr(outcome)
            logger.error(
                "Parallel subagent task failed before normalization: agent=%s error=%s",
                agent_id,
                exc_str,
            )
            normalized_results.append(
                SubagentRunOutcome(
                    agent_id=agent_id,
                    structured_result=SubagentResult(
                        status=SubagentStatus.FAILED,
                        summary="[error]",
                        errors=[f"agent {agent_id} failed"],
                    ),
                    execution_status=ExecutionStatus.FAILED,
                    error_message=str(outcome),
                    started_at=datetime.now(timezone.utc).isoformat(),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=0.0,
                    child_report=None,
                )
            )

        agent_ids = [result.agent_id for result in normalized_results]
        await self._emit(
            EV_SYNTHESIS_START,
            agent_ids=agent_ids,
            result_count=len(normalized_results),
        )
        for outcome in normalized_results:
            structured = outcome.structured_result
            await self._emit(
                EV_SYNTHESIS_INPUT_COLLECTED,
                child_agent_id=outcome.agent_id,
                status=(
                    structured.status.value
                    if structured is not None
                    else outcome.execution_status.value
                ),
                summary_preview=(
                    structured.summary[:300]
                    if structured is not None and structured.summary
                    else None
                ),
                tool_calls_made=(
                    outcome.child_report.tool_calls_total
                    if outcome.child_report is not None
                    else 0
                ),
                errors=(
                    list(structured.errors)
                    if structured is not None
                    else [outcome.error_message] if outcome.error_message else []
                ),
            )
        subagent_results = [result.structured_result for result in normalized_results]
        await self._emit(
            EV_SYNTHESIS_MERGE_START,
            agent_ids=agent_ids,
            merge_policy_name=type(self._policy).__name__,
        )
        merged = await self._policy.merge_results(subagent_results, agent_ids)
        await self._emit(
            EV_SYNTHESIS_MERGE_END,
            status=merged.status.value,
            merged_summary_preview=(merged.summary or "")[:600],
            artifacts_count=len(merged.artifacts),
            files_changed_count=len(merged.files_changed),
        )

        summaries: list[SubagentRunSummary] = []
        for outcome in normalized_results:
            if outcome.child_report is not None:
                summary = build_subagent_summary(outcome.child_report)
                if outcome.structured_result is not None:
                    normalized_status = (
                        outcome.execution_status
                        if outcome.execution_status != outcome.child_report.status
                        else (
                            execution_status_from_subagent_status(
                                outcome.structured_result.status
                            )
                            if outcome.child_report.status == ExecutionStatus.COMPLETED
                            else outcome.child_report.status
                        )
                    )
                    summary = summary.model_copy(
                        update={
                            "status": normalized_status,
                            "errors": (
                                []
                                if outcome.structured_result.status == SubagentStatus.SUCCESS
                                else outcome.structured_result.errors
                            ),
                        }
                    )
                summaries.append(summary)
                continue
            summaries.append(
                SubagentRunSummary(
                    agent_id=outcome.agent_id,
                    status=outcome.execution_status,
                    started_at=outcome.started_at,
                    finished_at=outcome.finished_at,
                    duration_ms=outcome.duration_ms,
                    errors=[outcome.error_message or outcome.execution_status.value],
                )
            )

        if report is not None:
            report.subagent_runs = summaries
            pending_approvals = [
                {
                    "agent_id": outcome.agent_id,
                    "plan": outcome.pending_shell_approval,
                }
                for outcome in normalized_results
                if isinstance(outcome.pending_shell_approval, dict)
            ]
            if pending_approvals:
                report.metadata["pending_shell_approvals"] = pending_approvals
            else:
                report.metadata.pop("pending_shell_approvals", None)
        await self._emit(
            EV_SYNTHESIS_END,
            status=merged.status.value,
            duration_ms=sum(item.duration_ms for item in normalized_results),
        )

        return merged, summaries


async def run_bypass(
    *,
    orchestrator: AgentOrchestrator,
    context: AgentContext,
    cancel_ctx: CancellationContext | None = None,
) -> tuple[Result, ExecutionReport]:
    """Fast-path: run a single subagent directly, skip leader planning."""
    if context.config.execution_mode != ExecutionMode.BYPASS:
        raise ContractViolationError(
            "BYPASS_MODE_REQUIRED",
            "run_bypass called but execution_mode is not BYPASS. "
            "Set execution_mode=ExecutionMode.BYPASS explicitly.",
        )
    return await orchestrator.run(
        context,
        run_kind=RunKind.SUBAGENT,
        cancel_ctx=cancel_ctx,
    )


__all__ = ["ParallelSubagentRunner", "run_bypass"]
