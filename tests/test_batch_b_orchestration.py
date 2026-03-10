"""Regression tests for orchestration and isolation (scoped event bus, etc.)."""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from protocore import (
    AgentConfig,
    AgentOrchestrator,
    CancellationContext,
    ContractViolationError,
    EventBus,
    ExecutionMode,
    ExecutionStatus,
    HookManager,
    Message,
    PathIsolationError,
    PolicyDecision,
    ShellToolConfig,
    ShellToolProfile,
    ToolContextMeta,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
    hookimpl,
    make_agent_context,
    make_execution_report,
)
from protocore.parallel import ParallelSubagentRunner
from protocore.orchestrator_errors import PendingShellApprovalError
from protocore.registry import AgentRegistry, ToolRegistry
from protocore.shell_handler import ShellHandler
from protocore.tool_dispatch import ToolDispatcher
from protocore.types import (
    ExecutionReport,
    Result,
    ShellCommandPlan,
    SubagentResult,
    SubagentStatus,
    ToolCall,
    ToolContext,
)
from protocore.orchestrator_utils import PolicyRunner


def _final_llm(content: str = "done") -> Any:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=Message(role="assistant", content=content))
    return llm


def _llm_tool_calls_then_final(
    tool_calls: list[ToolCall],
    *,
    final_content: str = "done",
) -> Any:
    call_count = 0
    llm = MagicMock()

    async def complete(*args: Any, **kwargs: Any) -> Message:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return Message(role="assistant", content=None, tool_calls=tool_calls)
        return Message(role="assistant", content=final_content)

    llm.complete = AsyncMock(side_effect=complete)
    return llm


def _tool_call(tool_name: str, arguments: dict[str, Any], *, call_id: str) -> ToolCall:
    return ToolCall.model_validate({
        "id": call_id,
        "function": {"name": tool_name, "arguments": arguments},
    })


def _report(context: Any) -> ExecutionReport:
    return make_execution_report(context=context)


class _AlwaysRetryPolicy:
    def should_retry(self, attempt: int, error: Exception) -> bool:
        _ = error
        return attempt < 3

    def delay_seconds(self, attempt: int) -> float:
        _ = attempt
        return 0.0


class TestScopedEventBus:
    @pytest.mark.asyncio
    async def test_concurrent_runs_do_not_mix_telemetry_by_request_id(self) -> None:
        seen_request_ids: dict[str, set[str]] = defaultdict(set)

        async def record_event(
            event_name: str,
            payload: dict[str, Any],
            report: ExecutionReport,
        ) -> None:
            _ = event_name
            seen_request_ids[report.request_id].add(str(payload.get("request_id")))

        collector = MagicMock()
        collector.record_event = AsyncMock(side_effect=record_event)

        async def complete(*args: Any, **kwargs: Any) -> Message:
            await asyncio.sleep(0.01)
            return Message(role="assistant", content="done")

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=complete)
        bus = EventBus()
        orch = AgentOrchestrator(
            llm_client=llm,
            event_bus=bus,
            telemetry_collector=collector,
        )

        ctx_a = make_agent_context(
            config=AgentConfig(agent_id="agent-a", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
        ctx_b = make_agent_context(
            config=AgentConfig(agent_id="agent-b", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
        ctx_a.messages.append(Message(role="user", content="task a"))
        ctx_b.messages.append(Message(role="user", content="task b"))

        await asyncio.gather(orch.run(ctx_a), orch.run(ctx_b))

        assert seen_request_ids[ctx_a.request_id] == {ctx_a.request_id}
        assert seen_request_ids[ctx_b.request_id] == {ctx_b.request_id}


class TestM010ChildContextRefs:
    def test_child_context_rewrites_request_and_metadata_refs(self) -> None:
        agent_registry = AgentRegistry()
        child_cfg = AgentConfig(
            agent_id="child",
            model="gpt-4o",
            execution_mode=ExecutionMode.BYPASS,
        )
        agent_registry.register(child_cfg)
        parent_cfg = AgentConfig(
            agent_id="parent",
            model="gpt-4o",
            execution_mode=ExecutionMode.PARALLEL,
        )
        parent = make_agent_context(config=parent_cfg)
        orch = AgentOrchestrator(llm_client=_final_llm(), agent_registry=agent_registry)

        child = orch._build_subagent_context(parent, "child", "do work")

        assert child.request_id != parent.request_id
        assert child.execution_metadata_ref == f"request:{child.request_id}:metadata"
        assert (
            child.tool_context.metadata[ToolContextMeta.REQUEST_ID] == child.request_id
        )
        assert (
            child.tool_context.metadata[ToolContextMeta.EXECUTION_METADATA_REF]
            == child.execution_metadata_ref
        )
        assert child.tool_context.agent_id == "child"

    @pytest.mark.asyncio
    async def test_child_run_events_are_forwarded_to_parent_bus(self) -> None:
        events: list[tuple[str, str | None, str | None, str | None]] = []

        async def _collect(event: Any) -> None:
            events.append(
                (
                    event.name,
                    event.payload.get("agent_id"),
                    event.payload.get("parent_agent_id"),
                    event.payload.get("run_kind"),
                )
            )

        bus = EventBus()
        bus.subscribe("*", _collect)
        agent_registry = AgentRegistry()
        agent_registry.register(
            AgentConfig(
                agent_id="child",
                model="gpt-4o",
                execution_mode=ExecutionMode.BYPASS,
            )
        )
        llm = _final_llm(
            '{"status":"success","summary":"child says hi","tool_calls_made":0}'
        )
        orch = AgentOrchestrator(
            llm_client=llm,
            event_bus=bus,
            agent_registry=agent_registry,
        )
        parent = make_agent_context(
            config=AgentConfig(
                agent_id="parent",
                model="gpt-4o",
                execution_mode=ExecutionMode.BYPASS,
            )
        )
        parent.messages.append(Message(role="user", content="delegate"))
        report = _report(parent)

        result = await orch._run_child_subagent(
            parent_context=parent,
            report=report,
            cancel_ctx=CancellationContext(),
            agent_id="child",
            task_text="do work",
        )

        assert result.content == "child says hi"
        assert ("session.start", "child", "parent", "subagent") in events
        assert ("llm.call.start", "child", "parent", "subagent") in events
        assert ("llm.call.end", "child", "parent", "subagent") in events
        assert ("session.end", "child", "parent", "subagent") in events


class TestM011StrictCloneIsolation:
    def test_uncloneable_tool_handler_blocks_child_spawn_in_strict_mode(self) -> None:
        class UncloneableHandler:
            def __deepcopy__(self, memo: dict[int, Any]) -> Any:
                _ = memo
                raise TypeError("cannot clone")

            async def __call__(
                self,
                *,
                arguments: dict[str, Any],
                context: ToolContext,
            ) -> ToolResult:
                _ = (arguments, context)
                return ToolResult(tool_call_id="tc1", tool_name="echo", content="ok")

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(name="echo", description="Echo tool"),
            UncloneableHandler(),
        )
        orch = AgentOrchestrator(llm_client=_final_llm(), tool_registry=registry)

        with pytest.raises(ContractViolationError) as exc_info:
            orch._spawn_child_orchestrator()
        assert exc_info.value.error_code == "TOOL_HANDLER_CLONE_REQUIRED"

    def test_uncloneable_hook_plugin_blocks_child_spawn_in_strict_mode(self) -> None:
        class UncloneablePlugin:
            def __deepcopy__(self, memo: dict[int, Any]) -> Any:
                _ = memo
                raise TypeError("cannot clone")

            @hookimpl
            def on_session_start(self, *, context: Any, report: Any) -> None:
                _ = (context, report)

        hooks = HookManager()
        hooks.register(UncloneablePlugin())
        orch = AgentOrchestrator(llm_client=_final_llm(), hook_manager=hooks)

        with pytest.raises(ContractViolationError) as exc_info:
            orch._spawn_child_orchestrator()
        assert exc_info.value.error_code == "HOOK_PLUGIN_CLONE_REQUIRED"


class TestM012ParallelCancellation:
    @pytest.mark.asyncio
    async def test_parent_cancellation_cancels_local_parallel_tasks(self) -> None:
        class SlowChildOrchestrator:
            async def run(
                self,
                context: Any,
                *,
                run_kind: Any,
                cancel_ctx: CancellationContext | None = None,
            ) -> tuple[Any, Any]:
                _ = (context, run_kind, cancel_ctx)
                await asyncio.sleep(60)
                raise AssertionError("should have been cancelled")

        class FakeParallelPolicy:
            max_concurrency = 1
            timeout_seconds = 60.0
            cancellation_mode = "propagate"

            async def merge_results(
                self,
                results: list[Any],
                agent_ids: list[str],
            ) -> SubagentResult:
                _ = (results, agent_ids)
                return SubagentResult(status=SubagentStatus.PARTIAL, summary="[cancelled]")

        ctx = make_agent_context(
            config=AgentConfig(agent_id="child", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
        runner = ParallelSubagentRunner(
            policy=FakeParallelPolicy(),
            orchestrator_factory=lambda: SlowChildOrchestrator(),
        )
        cancel_ctx = CancellationContext()

        async def trigger_cancel() -> None:
            await asyncio.sleep(0.05)
            cancel_ctx.cancel("stop")

        started = time.monotonic()
        runner_result, _ = await asyncio.gather(
            runner.run_parallel([("child", ctx)], cancel_ctx=cancel_ctx),
            trigger_cancel(),
        )
        _, summaries = runner_result
        elapsed = time.monotonic() - started

        assert elapsed < 1.0
        assert summaries[0].status == ExecutionStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_parallel_tool_dispatch_propagates_cancelled_error(self) -> None:
        registry = ToolRegistry()

        async def cancel_handler(*, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = (arguments, context)
            raise asyncio.CancelledError("stop")

        async def ok_handler(*, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = (arguments, context)
            return ToolResult(tool_call_id="tc2", tool_name="ok", content="ok")

        registry.register(ToolDefinition(name="cancel_me", description="cancel"), cancel_handler)
        registry.register(ToolDefinition(name="ok", description="ok"), ok_handler)
        llm = _llm_tool_calls_then_final(
            [
                _tool_call("cancel_me", {}, call_id="tc1"),
                _tool_call("ok", {}, call_id="tc2"),
            ]
        )
        orch = AgentOrchestrator(llm_client=llm, tool_registry=registry)
        ctx = make_agent_context(
            config=AgentConfig(
                agent_id="agent",
                model="gpt-4o",
                execution_mode=ExecutionMode.BYPASS,
                parallel_tool_calls=True,
                tool_definitions=[
                    ToolDefinition(name="cancel_me", description="cancel"),
                    ToolDefinition(name="ok", description="ok"),
                ],
            )
        )
        ctx.messages.append(Message(role="user", content="run tools"))

        result, report = await orch.run(ctx)

        assert result.status == ExecutionStatus.CANCELLED
        assert report.status == ExecutionStatus.CANCELLED


class TestM012bParallelNormalizationLogging:
    @pytest.mark.asyncio
    async def test_parallel_runner_logs_error_without_fake_traceback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class Policy:
            max_concurrency = 1
            timeout_seconds = 5.0
            cancellation_mode = "graceful"

            async def merge_results(
                self,
                results: list[Any],
                agent_ids: list[str],
            ) -> SubagentResult:
                _ = agent_ids
                return results[0] or SubagentResult(
                    status=SubagentStatus.FAILED,
                    summary="[missing]",
                )

        class FastChildOrchestrator:
            async def run(
                self,
                context: Any,
                *,
                run_kind: Any,
                cancel_ctx: CancellationContext | None = None,
            ) -> tuple[Any, Any]:
                _ = (context, run_kind, cancel_ctx)
                return (
                    MagicMock(content='{"status":"success","summary":"ok"}', artifacts=[]),
                    ExecutionReport(status=ExecutionStatus.COMPLETED, agent_id="child"),
                )

        async def fake_gather(*args: Any, **kwargs: Any) -> list[Any]:
            _ = (args, kwargs)
            return [RuntimeError("boom")]

        monkeypatch.setattr("protocore.parallel.asyncio.gather", fake_gather)

        cfg = AgentConfig(agent_id="child", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="work"))
        runner = ParallelSubagentRunner(
            policy=Policy(),
            orchestrator_factory=lambda: FastChildOrchestrator(),
        )

        with caplog.at_level("ERROR"):
            merged, summaries = await runner.run_parallel([(cfg.agent_id, ctx)])

        assert merged.status == SubagentStatus.FAILED
        assert summaries[0].status == ExecutionStatus.FAILED
        assert "before normalization: agent=child error=boom" in caplog.text
        assert "NoneType: None" not in caplog.text

    @pytest.mark.asyncio
    async def test_parallel_tool_dispatch_returns_partial_when_shell_approval_is_required(self) -> None:
        llm = _llm_tool_calls_then_final(
            [
                _tool_call("ok_tool", {}, call_id="tc1"),
                _tool_call("shell_exec", {"command": "rm -f tmp.txt"}, call_id="tc2"),
            ],
        )
        orch = AgentOrchestrator(llm_client=llm)
        ctx = make_agent_context(
            config=AgentConfig(
                agent_id="agent",
                model="gpt-4o",
                execution_mode=ExecutionMode.BYPASS,
                parallel_tool_calls=True,
                max_tool_calls=5,
                tool_definitions=[
                    ToolDefinition(name="ok_tool", description="ok"),
                    ToolDefinition(name="shell_exec", description="shell"),
                ],
            )
        )
        ctx.messages.append(Message(role="user", content="run tools"))

        async def dispatch_side_effect(
            tc: Any,
            context: Any,
            report: Any,
            run_kind: Any,
        ) -> ToolResult:
            _ = (context, report, run_kind)
            if tc.function.name == "ok_tool":
                return ToolResult(tool_call_id=tc.id, tool_name="ok_tool", content="ok")
            plan = ShellCommandPlan(
                plan_id="plan-approval",
                tool_call_id=tc.id,
                tool_name="shell_exec",
                command="rm -f tmp.txt",
            )
            raise PendingShellApprovalError(plan)

        orch._dispatch_tool = AsyncMock(side_effect=dispatch_side_effect)  # type: ignore[method-assign]

        result, report = await orch.run(ctx)

        assert result.status == ExecutionStatus.PARTIAL
        assert report.status == ExecutionStatus.PARTIAL
        assert result.content == "[approval required before shell execution]"
        assert result.metadata["pending_shell_approval"]["plan_id"] == "plan-approval"
        assert ctx.metadata["pending_shell_approval"]["plan_id"] == "plan-approval"
        assert any(msg.role == "tool" and msg.content == "ok" for msg in ctx.messages)

    @pytest.mark.asyncio
    async def test_parallel_tool_dispatch_handles_task_exception_and_forces_budget_finalization(self) -> None:
        llm = _llm_tool_calls_then_final(
            [
                _tool_call("explode", {}, call_id="tc1"),
                _tool_call("unused", {}, call_id="tc2"),
            ],
        )
        orch = AgentOrchestrator(llm_client=llm)
        ctx = make_agent_context(
            config=AgentConfig(
                agent_id="agent",
                model="gpt-4o",
                execution_mode=ExecutionMode.BYPASS,
                parallel_tool_calls=True,
                max_tool_calls=1,
                tool_definitions=[
                    ToolDefinition(name="explode", description="boom"),
                    ToolDefinition(name="unused", description="unused"),
                ],
            )
        )
        ctx.messages.append(Message(role="user", content="run tools"))

        async def dispatch_side_effect(
            tc: Any,
            context: Any,
            report: Any,
            run_kind: Any,
        ) -> ToolResult:
            _ = (context, report, run_kind)
            if tc.function.name == "explode":
                raise RuntimeError("boom")
            return ToolResult(tool_call_id=tc.id, tool_name=tc.function.name, content="ok")

        orch._dispatch_tool = AsyncMock(side_effect=dispatch_side_effect)  # type: ignore[method-assign]
        orch._finalize_after_tool_budget = AsyncMock(  # type: ignore[method-assign]
            return_value=Result(content="forced-final", status=ExecutionStatus.COMPLETED)
        )

        result, report = await orch.run(ctx)

        assert result.content == "forced-final"
        assert report.forced_finalization_triggered is True
        assert report.tool_calls_total == 1
        assert report.tool_failures == 1
        orch._finalize_after_tool_budget.assert_awaited()

    @pytest.mark.asyncio
    async def test_parallel_tool_dispatch_cancels_inflight_tasks_when_parent_is_cancelled(self) -> None:
        llm = _llm_tool_calls_then_final(
            [
                _tool_call("slow_one", {}, call_id="tc1"),
                _tool_call("slow_two", {}, call_id="tc2"),
            ],
        )
        orch = AgentOrchestrator(llm_client=llm)
        ctx = make_agent_context(
            config=AgentConfig(
                agent_id="agent",
                model="gpt-4o",
                execution_mode=ExecutionMode.BYPASS,
                parallel_tool_calls=True,
                max_tool_calls=5,
                tool_definitions=[
                    ToolDefinition(name="slow_one", description="slow"),
                    ToolDefinition(name="slow_two", description="slow"),
                ],
            )
        )
        ctx.messages.append(Message(role="user", content="run tools"))
        cancel_ctx = CancellationContext()

        cancelled_calls: list[str] = []

        async def dispatch_side_effect(
            tc: Any,
            context: Any,
            report: Any,
            run_kind: Any,
        ) -> ToolResult:
            _ = (context, report, run_kind)
            try:
                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                cancelled_calls.append(tc.id)
                raise
            return ToolResult(tool_call_id=tc.id, tool_name=tc.function.name, content="ok")

        orch._dispatch_tool = AsyncMock(side_effect=dispatch_side_effect)  # type: ignore[method-assign]

        async def trigger_cancel() -> None:
            await asyncio.sleep(0.03)
            cancel_ctx.cancel("user stop")

        (result, report), _ = await asyncio.gather(
            orch.run(ctx, cancel_ctx=cancel_ctx),
            trigger_cancel(),
        )

        assert result.status == ExecutionStatus.CANCELLED
        assert report.status == ExecutionStatus.CANCELLED
        assert set(cancelled_calls) == {"tc1", "tc2"}

    @pytest.mark.asyncio
    async def test_parallel_runner_propagates_pending_shell_approvals_to_report_metadata(
        self,
    ) -> None:
        class Policy:
            max_concurrency = 1
            timeout_seconds = 5.0
            cancellation_mode = "graceful"

            async def merge_results(
                self,
                results: list[Any],
                agent_ids: list[str],
            ) -> SubagentResult:
                _ = agent_ids
                return results[0] or SubagentResult(
                    status=SubagentStatus.PARTIAL,
                    summary="[approval required]",
                    errors=["APPROVAL_REQUIRED"],
                )

        class ChildOrchestrator:
            async def run(
                self,
                context: Any,
                *,
                run_kind: Any,
                cancel_ctx: CancellationContext | None = None,
            ) -> tuple[Any, Any]:
                _ = (run_kind, cancel_ctx)
                return (
                    SimpleNamespace(
                        content="[approval required before shell execution]",
                        artifacts=[],
                        metadata={
                            "pending_shell_approval": {
                                "plan_id": "plan-1",
                                "command": "rm ./tmp.txt",
                                "approval_status": "pending",
                            }
                        },
                    ),
                    ExecutionReport(
                        status=ExecutionStatus.PARTIAL,
                        error_message="APPROVAL_REQUIRED",
                        agent_id=context.config.agent_id,
                    ),
                )

        cfg = AgentConfig(agent_id="child", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="work"))
        runner = ParallelSubagentRunner(
            policy=Policy(),
            orchestrator_factory=lambda: ChildOrchestrator(),
        )
        report = ExecutionReport()

        _merged, _summaries = await runner.run_parallel([(cfg.agent_id, ctx)], report=report)

        pending = report.metadata.get("pending_shell_approvals")
        assert isinstance(pending, list)
        assert pending[0]["agent_id"] == "child"
        assert pending[0]["plan"]["plan_id"] == "plan-1"


class TestM013IdempotentRetry:
    @pytest.mark.asyncio
    async def test_non_idempotent_tool_is_not_retried(self) -> None:
        llm = _llm_tool_calls_then_final(
            [_tool_call("mutate", {}, call_id="tc1")],
        )
        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=RuntimeError("transient"))
        orch = AgentOrchestrator(
            llm_client=llm,
            tool_executor=executor,
            retry_policy=_AlwaysRetryPolicy(),
        )
        ctx = make_agent_context(
            config=AgentConfig(
                agent_id="agent",
                model="gpt-4o",
                execution_mode=ExecutionMode.BYPASS,
                tool_definitions=[
                    ToolDefinition(
                        name="mutate",
                        description="Mutating tool",
                        idempotent=False,
                    )
                ],
            )
        )
        ctx.messages.append(Message(role="user", content="run mutate"))

        result, report = await orch.run(ctx)

        assert result.status == ExecutionStatus.COMPLETED
        assert report.status == ExecutionStatus.COMPLETED
        assert executor.execute.await_count == 1
        assert not any("retry:tool.executor_execute" in warning for warning in report.warnings)

    @pytest.mark.asyncio
    async def test_idempotent_tool_can_be_retried(self) -> None:
        llm = _llm_tool_calls_then_final(
            [_tool_call("read_status", {}, call_id="tc1")],
        )
        executor = MagicMock()
        executor.execute = AsyncMock(
            side_effect=[
                RuntimeError("transient"),
                ToolResult(tool_call_id="tc1", tool_name="read_status", content="ok"),
            ]
        )
        orch = AgentOrchestrator(
            llm_client=llm,
            tool_executor=executor,
            retry_policy=_AlwaysRetryPolicy(),
        )
        ctx = make_agent_context(
            config=AgentConfig(
                agent_id="agent",
                model="gpt-4o",
                execution_mode=ExecutionMode.BYPASS,
                tool_definitions=[
                    ToolDefinition(
                        name="read_status",
                        description="Read-only tool",
                        idempotent=True,
                    )
                ],
            )
        )
        ctx.messages.append(Message(role="user", content="read status"))

        result, report = await orch.run(ctx)

        assert result.status == ExecutionStatus.COMPLETED
        assert executor.execute.await_count == 2
        assert any("retry:tool.executor_execute" in warning for warning in report.warnings)


class TestM014DispatchParity:
    @pytest.mark.asyncio
    async def test_validation_errors_match_in_serial_and_parallel_dispatch(self, tmp_path: Path) -> None:
        tool_calls = [
            _tool_call("write_file", {"target_directory": "/etc"}, call_id="tc1"),
            _tool_call("write_file", {"target_directory": "/etc"}, call_id="tc2"),
        ]
        tool_def = ToolDefinition(
            name="write_file",
            description="Writes files",
            filesystem_access=True,
            path_fields=["target_directory"],
        )

        async def run_case(parallel_tool_calls: bool) -> tuple[Any, Any, Any]:
            llm = _llm_tool_calls_then_final(tool_calls)
            executor = MagicMock()
            executor.execute = AsyncMock()
            orch = AgentOrchestrator(llm_client=llm, tool_executor=executor)
            ctx = make_agent_context(
                config=AgentConfig(
                    agent_id="agent",
                    model="gpt-4o",
                    execution_mode=ExecutionMode.BYPASS,
                    parallel_tool_calls=parallel_tool_calls,
                    tool_definitions=[tool_def],
                ),
                allowed_paths=[str(tmp_path)],
            )
            ctx.messages.append(Message(role="user", content="write files"))
            result, report = await orch.run(ctx)
            tool_messages = [msg.content for msg in ctx.messages if msg.role == "tool"]
            return result, report, tool_messages

        serial_result, serial_report, serial_messages = await run_case(False)
        parallel_result, parallel_report, parallel_messages = await run_case(True)

        assert serial_result.status == ExecutionStatus.COMPLETED
        assert parallel_result.status == ExecutionStatus.COMPLETED
        assert serial_report.tool_failures == 2
        assert parallel_report.tool_failures == 2
        assert serial_messages == parallel_messages
        assert all(
            isinstance(message, str)
            and message.startswith("[TOOL ERROR]\n[TOOL VALIDATION ERROR:")
            for message in serial_messages
        )


class TestM015VisibleToolAllowlist:
    @pytest.mark.asyncio
    async def test_empty_visible_tool_list_is_deny_all(self) -> None:
        dispatcher = ToolDispatcher(
            hooks=HookManager(),
            event_bus=EventBus(),
            policy=None,
            tool_registry=None,
            tool_executor=None,
            shell_executor=None,
            shell_safety_policy=None,
            shell_handler=ShellHandler(
                shell_executor=None,
                policy_runner=PolicyRunner(),
                shell_safety_policy=None,
                append_tool_results_as_messages=lambda messages, results: None,
            ),
            policy_runner=PolicyRunner(),
        )
        ctx = make_agent_context(
            config=AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
        ctx.tool_context.metadata[ToolContextMeta.VISIBLE_TOOL_NAMES] = []

        decision, source = await dispatcher.evaluate_tool_preflight(
            tool_name="echo",
            arguments={},
            context=ctx,
            report=_report(ctx),
            shell_capability=None,
            shell_tool_name="shell_exec",
        )

        assert decision == PolicyDecision.DENY
        assert source == "tool_visibility"


class TestM016RelativeCwdResolution:
    def test_relative_cwd_is_resolved_inside_workspace(self, tmp_path: Path) -> None:
        tool_context = ToolContext(
            session_id="s1",
            trace_id="t1",
            agent_id="a1",
            allowed_paths=[str(tmp_path)],
            metadata={ToolContextMeta.WORKSPACE_ROOT: str(tmp_path)},
        )
        capability = ShellToolConfig(profile=ShellToolProfile.WORKSPACE_WRITE)

        request = ShellHandler.normalize_shell_request(
            {"command": "pwd", "cwd": "subdir"},
            capability,
            tool_context,
        )

        assert request.cwd == str((tmp_path / "subdir").resolve())

    def test_relative_cwd_escape_is_blocked(self, tmp_path: Path) -> None:
        tool_context = ToolContext(
            session_id="s1",
            trace_id="t1",
            agent_id="a1",
            allowed_paths=[str(tmp_path)],
            metadata={ToolContextMeta.WORKSPACE_ROOT: str(tmp_path)},
        )
        capability = ShellToolConfig(profile=ShellToolProfile.WORKSPACE_WRITE)

        with pytest.raises(PathIsolationError):
            ShellHandler.normalize_shell_request(
                {"command": "pwd", "cwd": "../escape"},
                capability,
                tool_context,
            )


class TestM017SchemaDrivenPathIsolation:
    def test_filesystem_tool_without_path_annotations_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolDefinition(
                name="download",
                description="Downloads a file",
                filesystem_access=True,
                parameters=ToolParameterSchema(
                    properties={"target_directory": {"type": "string"}},
                ),
            )

    @pytest.mark.asyncio
    async def test_custom_path_field_is_validated_when_declared(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        handler = AsyncMock(
            return_value=ToolResult(tool_call_id="tc1", tool_name="download", content="ok")
        )
        registry.register(
            ToolDefinition(
                name="download",
                description="Downloads a file",
                filesystem_access=True,
                path_fields=["target_directory"],
                parameters=ToolParameterSchema(
                    properties={"target_directory": {"type": "string"}},
                ),
            ),
            handler,
        )
        ctx = ToolContext(
            session_id="s1",
            trace_id="t1",
            agent_id="a1",
            allowed_paths=[str(tmp_path)],
            metadata={ToolContextMeta.WORKSPACE_ROOT: str(tmp_path)},
        )

        with pytest.raises(PathIsolationError):
            await registry.dispatch(
                "download",
                {"target_directory": "/etc"},
                ctx,
            )

        assert handler.await_count == 0


class TestM018HookStringDecision:
    @pytest.mark.asyncio
    async def test_hook_deny_string_stops_tool_before_handler(self) -> None:
        class DenyPlugin:
            @hookimpl
            def on_tool_pre_execute(
                self,
                tool_name: str,
                arguments: dict[str, Any],
                context: Any,
                report: Any,
            ) -> str:
                _ = (tool_name, arguments, context, report)
                return " deny "

        hooks = HookManager()
        hooks.register(DenyPlugin())
        registry = ToolRegistry()
        handler = AsyncMock(
            return_value=ToolResult(tool_call_id="tc1", tool_name="dangerous", content="ok")
        )
        registry.register(
            ToolDefinition(name="dangerous", description="Dangerous tool"),
            handler,
        )
        llm = _llm_tool_calls_then_final(
            [_tool_call("dangerous", {}, call_id="tc1")],
        )
        orch = AgentOrchestrator(
            llm_client=llm,
            tool_registry=registry,
            hook_manager=hooks,
        )
        ctx = make_agent_context(
            config=AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
        ctx.messages.append(Message(role="user", content="run dangerous"))

        result, report = await orch.run(ctx)

        assert result.status == ExecutionStatus.COMPLETED
        assert report.tool_failures == 1
        assert handler.await_count == 0
        tool_messages = [msg.content for msg in ctx.messages if msg.role == "tool"]
        assert tool_messages == ["[TOOL ERROR]\n[DENIED by hook preflight: dangerous]"]
