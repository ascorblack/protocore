from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from protocore import (
    AgentConfig,
    AgentOrchestrator,
    ContractViolationError,
    AgentRegistry,
    AgentRole,
    AgentEnvelope,
    CancellationContext,
    CompactionSummary,
    ControlCommand,
    EventBus,
    ExecutionMode,
    ExecutionStatus,
    EV_DESTRUCTIVE_ACTION,
    EV_INJECTION_SIGNAL,
    HookManager,
    Message,
    PlanArtifact,
    RunKind,
    ShellAccessMode,
    ShellExecutionResult,
    ShellToolConfig,
    SubagentResult,
    SubagentStatus,
    ToolDefinition,
    LLMUsage,
    ToolResult,
    ToolRegistry,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowNode,
    hookimpl,
    make_agent_context,
    make_execution_report,
)
from protocore.constants import THINKING_PROFILE_DEFAULTS, ThinkingProfileRegistry
from protocore.events import EV_LLM_CALL_FAILED, EV_LLM_STREAM_DELTA
from protocore.orchestrator import ParallelSubagentRunner
from protocore.orchestrator_utils import PolicyRunner
from protocore.shell_handler import ShellHandler
from protocore.tool_dispatch import ToolDispatcher
from protocore.types import MessageList, ToolCall

# Portable path for shell tests
_TEST_CWD = str(Path(tempfile.gettempdir()) / "protocore-test")


def _final_llm(content: str = "done") -> Any:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=Message(role="assistant", content=content))
    return llm


class TestRemediationContracts:
    @pytest.mark.asyncio
    async def test_event_bus_emits_envelope_fields(self) -> None:
        bus = EventBus()
        seen: list[Any] = []

        async def handler(event: Any) -> None:
            seen.append(event)

        bus.subscribe("session.start", handler)
        await bus.emit_simple(
            "session.start",
            agent_id="leader",
            request_id="req-1",
            trace_id="trace-1",
            session_id="session-1",
            parent_agent_id=None,
            run_kind="leader",
            execution_mode="bypass",
            phase="main_turn",
        )

        assert len(seen) == 1
        event = seen[0]
        assert event.ts > 0
        assert event.seq == 1
        assert event.run_id == "req-1"
        assert event.payload["phase"] == "main_turn"

    @pytest.mark.asyncio
    async def test_subagent_bus_forwards_events_to_parent_bus_unchanged(self) -> None:
        registry = AgentRegistry()
        registry.register(
            AgentConfig(
                agent_id="child-1",
                model="gpt-4o",
                role=AgentRole.SUBAGENT,
                execution_mode=ExecutionMode.BYPASS,
            )
        )

        selection = MagicMock()
        selection.select = AsyncMock(return_value="child-1")

        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(role="assistant", content="ok"))

        bus = EventBus()
        seen: list[Any] = []

        async def on_stream(event: Any) -> None:
            if event.payload.get("run_kind") == "subagent":
                seen.append(event)

        bus.subscribe("session.start", on_stream)

        ctx = make_agent_context(
            config=AgentConfig(
                agent_id="leader",
                model="gpt-4o",
                role=AgentRole.LEADER,
                execution_mode=ExecutionMode.AUTO_SELECT,
            )
        )
        ctx.messages.append(Message(role="user", content="delegate"))

        orch = AgentOrchestrator(
            llm_client=llm,
            event_bus=bus,
            agent_registry=registry,
            subagent_selection_policy=selection,
        )
        await orch.run(ctx, run_kind=RunKind.LEADER)

        assert seen
        child_event = seen[0]
        assert child_event.payload["agent_id"] == "child-1"
        assert child_event.payload["parent_agent_id"] == "leader"
        assert child_event.payload["run_kind"] == "subagent"

    def test_contract_violation_error_is_exposed_in_public_api(self) -> None:
        from protocore import ContractViolationError

        err = ContractViolationError("CODE", "message")
        assert err.error_code == "CODE"

    def test_safety_events_exposed_in_public_api(self) -> None:
        assert EV_DESTRUCTIVE_ACTION == "safety.destructive_action"
        assert EV_INJECTION_SIGNAL == "safety.injection_signal"

    @pytest.mark.asyncio
    async def test_session_start_hook_failure_still_returns_report(self) -> None:
        class BrokenPlugin:
            @hookimpl
            def on_session_start(self, context: Any, report: Any) -> None:
                raise RuntimeError("hook failed")

        hooks = HookManager()
        hooks.register(BrokenPlugin())

        cfg = AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=_final_llm(), hook_manager=hooks)
        result, report = await orch.run(ctx)

        assert result.status == ExecutionStatus.COMPLETED
        assert report.status == ExecutionStatus.COMPLETED
        assert "hook_failed:on_session_start:RuntimeError" in report.warnings

    def test_thinking_profile_defaults_view_tracks_runtime_mutations(self) -> None:
        assert "temporary_runtime_profile" not in THINKING_PROFILE_DEFAULTS

        ThinkingProfileRegistry.register(
            "temporary_runtime_profile",
            {"enable_thinking": True},
        )
        try:
            assert "temporary_runtime_profile" in ThinkingProfileRegistry.all_profiles()
            assert "temporary_runtime_profile" in THINKING_PROFILE_DEFAULTS
        finally:
            ThinkingProfileRegistry.unregister("temporary_runtime_profile")

    @pytest.mark.asyncio
    async def test_auto_select_subagent_result_uses_structured_fallback(self) -> None:
        selection = MagicMock()
        selection.select = AsyncMock(return_value="child-1")

        registry = AgentRegistry()
        registry.register(
            AgentConfig(
                agent_id="child-1",
                model="gpt-4o",
                role=AgentRole.SUBAGENT,
                execution_mode=ExecutionMode.LEADER,
            )
        )

        parent = AgentConfig(
            agent_id="leader",
            model="gpt-4o",
            role=AgentRole.LEADER,
            execution_mode=ExecutionMode.AUTO_SELECT,
        )
        ctx = make_agent_context(config=parent)
        ctx.messages.append(Message(role="user", content="delegate this"))

        orch = AgentOrchestrator(
            llm_client=_final_llm("not valid json"),
            subagent_selection_policy=selection,
            agent_registry=registry,
        )
        result, report = await orch.run(ctx, run_kind=RunKind.LEADER)

        assert result.status == ExecutionStatus.PARTIAL
        assert report.status == ExecutionStatus.PARTIAL
        assert result.content == "not valid json"
        assert "subagent_result_fallback:child-1" in report.warnings

    @pytest.mark.asyncio
    async def test_cancel_propagation_failure_does_not_break_cancelled_report(self) -> None:
        transport = MagicMock()
        transport.send = AsyncMock(side_effect=RuntimeError("transport down"))

        cfg = AgentConfig(agent_id="leader", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="cancel me"))
        ctx.metadata["_active_child_agent_ids"] = ["child-a"]

        cancel_ctx = CancellationContext()
        cancel_ctx.cancel("stop")

        orch = AgentOrchestrator(
            llm_client=_final_llm(),
            transport=transport,
        )
        result, report = await orch.run(ctx, cancel_ctx=cancel_ctx)

        assert result.status == ExecutionStatus.CANCELLED
        assert report.status == ExecutionStatus.CANCELLED
        assert any(w.startswith("cancel_propagation_failed:child-a") for w in report.warnings)

    @pytest.mark.asyncio
    async def test_executor_only_filesystem_tool_is_validated_before_execute(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value=Message(
                role="assistant",
                tool_calls=[
                    ToolCall.model_validate({
                        "id": "tc1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/etc/passwd"}',
                        },
                    })
                ],
            )
        )
        executor = MagicMock()
        executor.execute = AsyncMock(
            return_value=ToolResult(tool_call_id="tc1", tool_name="read_file", content="nope")
        )

        cfg = AgentConfig(
            agent_id="agent",
            model="gpt-4o",
            execution_mode=ExecutionMode.BYPASS,
            tool_definitions=[
                ToolDefinition(
                    name="read_file",
                    description="Read file",
                    filesystem_access=True,
                    path_fields=["path"],
                )
            ],
        )
        ctx = make_agent_context(config=cfg, allowed_paths=[])
        ctx.messages.append(Message(role="user", content="read secret"))

        orch = AgentOrchestrator(
            llm_client=llm,
            tool_executor=executor,
        )
        result, report = await orch.run(ctx)

        assert result.status == ExecutionStatus.COMPLETED
        assert report.tool_failures >= 1
        executor.execute.assert_not_awaited()

    def test_workflow_definition_rejects_duplicate_node_ids(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowDefinition(
                nodes=[
                    WorkflowNode(node_id="n1", label="a"),
                    WorkflowNode(node_id="n1", label="b"),
                ]
            )

    def test_workflow_definition_rejects_cycles(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowDefinition(
                nodes=[
                    WorkflowNode(node_id="n1", label="a"),
                    WorkflowNode(node_id="n2", label="b"),
                ],
                edges=[
                    WorkflowEdge(from_node="n1", to_node="n2"),
                    WorkflowEdge(from_node="n2", to_node="n1"),
                ],
            )

    @pytest.mark.asyncio
    async def test_plan_artifact_is_persisted_in_execution_report(self) -> None:
        plan = PlanArtifact(trace_id="trace-1", raw_plan="step 1")
        planning = MagicMock()
        planning.build_plan = AsyncMock(return_value=plan)

        cfg = AgentConfig(agent_id="leader", model="gpt-4o", execution_mode=ExecutionMode.LEADER)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="plan this"))

        orch = AgentOrchestrator(
            llm_client=_final_llm(),
            planning_strategy=planning,
        )
        result, report = await orch.run(ctx, run_kind=RunKind.LEADER)

        assert result.status == ExecutionStatus.COMPLETED
        assert report.plan_artifact is not None
        assert report.plan_artifact.plan_id == plan.plan_id

    @pytest.mark.asyncio
    async def test_telemetry_collector_receives_runtime_events(self) -> None:
        collector = MagicMock()
        collector.record_event = AsyncMock()
        bus = EventBus()

        cfg = AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(
            llm_client=_final_llm(),
            event_bus=bus,
            telemetry_collector=collector,
        )
        result, report = await orch.run(ctx)

        assert result.status == ExecutionStatus.COMPLETED
        assert report.status == ExecutionStatus.COMPLETED
        event_names = [call.args[0] for call in collector.record_event.await_args_list]
        assert "session.start" in event_names
        assert "session.end" in event_names

    @pytest.mark.asyncio
    async def test_run_waits_for_background_telemetry_flush(self) -> None:
        seen: list[str] = []

        async def record_event(event_name: str, payload: Any, report: Any) -> None:
            _ = (payload, report)
            await asyncio.sleep(0.01)
            seen.append(event_name)

        collector = MagicMock()
        collector.record_event = AsyncMock(side_effect=record_event)
        bus = EventBus()

        cfg = AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(
            llm_client=_final_llm(),
            event_bus=bus,
            telemetry_collector=collector,
        )
        result, report = await orch.run(ctx)

        assert result.status == ExecutionStatus.COMPLETED
        assert report.status == ExecutionStatus.COMPLETED
        assert "session.start" in seen
        assert "session.end" in seen

    def test_parse_envelope_with_report_records_minor_warning(self) -> None:
        report = make_execution_report(
            context=make_agent_context(
                config=AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
            )
        )
        payload = {
            "protocol_version": "1.1",
            "message_type": "task",
            "trace_id": "t",
            "session_id": "s",
            "sender": {"agent_id": "leader", "role": "leader"},
            "recipient": {"agent_id": "sub", "role": "subagent"},
            "payload": {"task": "hello"},
            "meta": {"created_at": "2026-03-07T00:00:00+00:00", "protocol_version": "1.1"},
        }

        envelope = AgentEnvelope.parse_with_report(payload, report)

        assert envelope.protocol_version == "1.1"
        assert any(w.startswith("protocol_minor_version_mismatch") for w in report.warnings)

    @pytest.mark.asyncio
    async def test_parallel_runner_records_non_zero_duration(self) -> None:
        class Policy:
            max_concurrency = 1
            timeout_seconds = 5.0
            cancellation_mode = "graceful"

            async def merge_results(
                self,
                results: list[SubagentResult | None],
                agent_ids: list[str],
            ) -> SubagentResult:
                return results[0] or SubagentResult(status=SubagentStatus.FAILED, summary="missing")

        async def delayed_complete(*args: Any, **kwargs: Any) -> Message:
            await asyncio.sleep(0.01)
            return Message(
                role="assistant",
                content=(
                    '{"status":"success","summary":"ok","artifacts":[],"files_changed":[],'
                    '"tool_calls_made":0,"errors":[],"next_steps":null}'
                ),
            )

        def make_orchestrator() -> AgentOrchestrator:
            llm = MagicMock()
            llm.complete = AsyncMock(side_effect=delayed_complete)
            return AgentOrchestrator(llm_client=llm)

        cfg = AgentConfig(agent_id="child", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="work"))

        runner = ParallelSubagentRunner(policy=Policy(), orchestrator_factory=make_orchestrator)
        merged, summaries = await runner.run_parallel([(cfg.agent_id, ctx)])

        assert merged.status == "success"
        assert len(summaries) == 1
        assert summaries[0].duration_ms > 0

    @pytest.mark.asyncio
    async def test_parallel_mode_rejects_excessive_agent_count(self) -> None:
        policy = MagicMock(
            max_concurrency=2,
            timeout_seconds=1.0,
            cancellation_mode="graceful",
            merge_results=AsyncMock(
                return_value=SubagentResult(status=SubagentStatus.SUCCESS, summary="merged")
            ),
        )
        orch = AgentOrchestrator(
            llm_client=_final_llm(),
            parallel_execution_policy=policy,
        )
        ctx = make_agent_context(
            config=AgentConfig(
                agent_id="leader",
                model="gpt-4o",
                execution_mode=ExecutionMode.PARALLEL,
            )
        )
        ctx.messages.append(Message(role="user", content="fan out"))
        ctx.metadata["parallel_agent_ids"] = [f"agent-{idx}" for idx in range(51)]

        _, report = await orch.run(ctx)

        assert report.status == ExecutionStatus.FAILED
        assert report.error_code == "PARALLEL_AGENT_IDS_TOO_MANY"

    @pytest.mark.asyncio
    async def test_parallel_runner_rejects_excessive_concurrency(self) -> None:
        class Policy:
            max_concurrency = 101
            timeout_seconds = 1.0
            cancellation_mode = "graceful"

            async def merge_results(
                self,
                results: list[SubagentResult | None],
                agent_ids: list[str],
            ) -> SubagentResult:
                _ = (results, agent_ids)
                return SubagentResult(status=SubagentStatus.SUCCESS, summary="merged")

        runner = ParallelSubagentRunner(
            policy=Policy(),
            orchestrator_factory=lambda: AgentOrchestrator(llm_client=_final_llm()),
        )

        cfg = AgentConfig(agent_id="child", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="work"))

        with pytest.raises(ContractViolationError, match="safe limit"):
            await runner.run_parallel([(cfg.agent_id, ctx)])

    @pytest.mark.asyncio
    async def test_handle_error_hides_exception_details_from_result_content(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("secret-token=123"))

        cfg = AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="trigger error"))

        orch = AgentOrchestrator(llm_client=llm)
        result, report = await orch.run(ctx)

        assert result.status == ExecutionStatus.FAILED
        assert result.content.startswith("[error:")
        assert "secret-token=123" not in result.content
        assert report.error_message is not None

    def test_shell_request_rejects_reserved_and_oversized_env_values(self) -> None:
        from protocore.context import build_tool_context

        capability = ShellToolConfig(
            access_mode=ShellAccessMode.LEADER_ONLY,
            tool_name="shell_exec",
            env_allowlist=["SAFE_ENV", "PATH"],
        )
        tool_context = build_tool_context(
            agent_id="agent",
            session_id="session",
            trace_id="trace",
            allowed_paths=["/tmp"],
        )

        with pytest.raises(ValueError, match="reserved"):
            AgentOrchestrator._normalize_shell_request(
                {"command": "echo ok", "env": {"PATH": "/tmp/evil"}},
                capability,
                tool_context,
            )

        with pytest.raises(ValueError, match="max length"):
            AgentOrchestrator._normalize_shell_request(
                {"command": "echo ok", "env": {"SAFE_ENV": "x" * 5000}},
                capability,
                tool_context,
            )

    @pytest.mark.asyncio
    async def test_session_end_hook_failure_is_reported_as_warning(self) -> None:
        class BrokenPlugin:
            @hookimpl
            def on_session_end(self, context: Any, report: Any) -> None:
                raise RuntimeError("session end failed")

        hooks = HookManager()
        hooks.register(BrokenPlugin())

        cfg = AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=_final_llm(), hook_manager=hooks)
        result, report = await orch.run(ctx)

        assert result.status == ExecutionStatus.COMPLETED
        assert "hook_failed:on_session_end:RuntimeError" in report.warnings

    @pytest.mark.asyncio
    async def test_error_hook_failure_is_reported_and_llm_call_failed_event_is_emitted(self) -> None:
        hooks = MagicMock(spec=HookManager)
        hooks.call_on_error.side_effect = RuntimeError("error hook failed")
        hooks.call_on_cancelled.return_value = None

        bus = EventBus()
        seen: list[Any] = []

        async def on_llm_call_failed(event: Any) -> None:
            seen.append(event)

        bus.subscribe(EV_LLM_CALL_FAILED, on_llm_call_failed)

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("boom"))

        cfg = AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="trigger error"))

        orch = AgentOrchestrator(
            llm_client=llm,
            hook_manager=hooks,
            event_bus=bus,
        )
        result, report = await orch.run(ctx)

        assert result.status == ExecutionStatus.FAILED
        assert any(w.startswith("error_hook_failed:RuntimeError") for w in report.warnings)
        assert len(seen) == 1
        assert seen[0].payload["error_code"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_cancelled_hook_failure_is_reported_as_warning(self) -> None:
        hooks = MagicMock(spec=HookManager)
        hooks.call_on_cancelled.side_effect = RuntimeError("cancelled hook failed")
        hooks.call_on_error.return_value = None

        cfg = AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="cancel"))

        cancel_ctx = CancellationContext()
        cancel_ctx.cancel("manual")

        orch = AgentOrchestrator(
            llm_client=_final_llm(),
            hook_manager=hooks,
        )
        result, report = await orch.run(ctx, cancel_ctx=cancel_ctx)

        assert result.status == ExecutionStatus.CANCELLED
        assert report.status == ExecutionStatus.CANCELLED
        assert any(w.startswith("cancelled_hook_failed:RuntimeError") for w in report.warnings)

    def test_public_error_reason_prefers_timeout_contract_and_status_code(self) -> None:
        class UpstreamFailure(RuntimeError):
            status_code = 503

        contract_error = ContractViolationError("BAD_INPUT", "bad input")

        assert AgentOrchestrator._public_error_reason(exc=TimeoutError("x"), is_timeout=True) == "timeout"
        assert (
            AgentOrchestrator._public_error_reason(exc=contract_error, is_timeout=False)
            == "contract:bad_input"
        )
        assert (
            AgentOrchestrator._public_error_reason(exc=UpstreamFailure("x"), is_timeout=False)
            == "upstream 503"
        )

    @pytest.mark.asyncio
    async def test_stream_event_callback_ignores_empty_text_and_emits_reasoning_delta(self) -> None:
        bus = EventBus()
        captured: list[Any] = []

        async def on_delta(event: Any) -> None:
            captured.append(event)

        bus.subscribe(EV_LLM_STREAM_DELTA, on_delta)

        cfg = AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)

        orch = AgentOrchestrator(llm_client=_final_llm(), event_bus=bus)
        callback, stats = orch._make_stream_event_callback(context=ctx, iteration=1)

        await callback({"kind": "text", "text": ""})
        await callback(
            {
                "kind": "reasoning",
                "text": "abc",
                "provider_event_type": "response.reasoning.delta",
            }
        )

        assert len(captured) == 1
        assert captured[0].payload["kind"] == "reasoning"
        assert stats["delta_count"] == 1
        assert stats["reasoning_chars"] == 3

    @pytest.mark.asyncio
    async def test_queue_wait_metric_is_loaded_from_context_metadata(self) -> None:
        cfg = AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg, metadata={"queue_wait_ms": 12.5})
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=_final_llm())
        _, report = await orch.run(ctx)

        assert report.queue_wait_ms == 12.5

    @pytest.mark.asyncio
    async def test_queue_wait_invalid_value_adds_warning(self) -> None:
        cfg = AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg, metadata={"queue_wait_ms": "not-a-number"})
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=_final_llm())
        _, report = await orch.run(ctx)

        assert report.queue_wait_ms is None
        assert "queue_wait_ms_invalid" in report.warnings

    @pytest.mark.asyncio
    async def test_estimated_cost_is_accumulated_from_usage(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value=Message(
                role="assistant",
                content="done",
                usage=LLMUsage(input_tokens=100, output_tokens=50, cached_tokens=0),
            )
        )
        cfg = AgentConfig(
            agent_id="agent",
            model="gpt-4o",
            execution_mode=ExecutionMode.BYPASS,
            cost_per_token=0.001,
        )
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=llm)
        _, report = await orch.run(ctx)

        assert report.estimated_cost == pytest.approx(0.15)

    @pytest.mark.asyncio
    async def test_tool_budget_finalization_tracks_second_llm_latency(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(
            side_effect=[
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall.model_validate(
                            {"id": "tc-1", "function": {"name": "echo", "arguments": "{}"}}
                        ),
                        ToolCall.model_validate(
                            {"id": "tc-2", "function": {"name": "echo", "arguments": "{}"}}
                        ),
                    ],
                ),
                Message(role="assistant", content="final answer"),
            ]
        )
        cfg = AgentConfig(
            agent_id="agent",
            model="gpt-4o",
            execution_mode=ExecutionMode.BYPASS,
            max_tool_calls=1,
            tool_definitions=[ToolDefinition(name="echo", description="Echo tool")],
        )
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="run tools"))

        orch = AgentOrchestrator(llm_client=llm)
        _, report = await orch.run(ctx)

        assert report.forced_finalization_triggered is True
        assert len(report.llm_latency_ms) == 2

    @pytest.mark.asyncio
    async def test_tool_post_execute_hook_failure_is_reported_as_warning(self) -> None:
        class BrokenToolHook:
            @hookimpl
            def on_tool_post_execute(self, result: Any, context: Any, report: Any) -> None:
                _ = (result, context, report)
                raise RuntimeError("boom")

        hooks = HookManager()
        hooks.register(BrokenToolHook())
        llm = MagicMock()
        llm.complete = AsyncMock(
            side_effect=[
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall.model_validate(
                            {
                                "id": "tc-echo",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"text":"hi"}',
                                },
                            }
                        )
                    ],
                ),
                Message(role="assistant", content="ok"),
            ]
        )
        orch = AgentOrchestrator(
            llm_client=llm,
            hook_manager=hooks,
        )
        orch._tool_registry.register(
            ToolDefinition(name="echo", description="echo"),
            AsyncMock(return_value=ToolResult(tool_call_id="tc-echo", tool_name="echo", content="ok")),
        )
        ctx = make_agent_context(
            config=AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
        ctx.messages.append(Message(role="user", content="run echo"))
        _result, report = await orch.run(ctx)
        assert "hook_failed:on_tool_post_execute:RuntimeError" in report.warnings

    @pytest.mark.asyncio
    async def test_shell_result_risk_flags_are_added_to_report(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(
            side_effect=[
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall.model_validate(
                            {
                                "id": "tc-shell",
                                "function": {
                                    "name": "shell_exec",
                                    "arguments": f'{{"command":"pwd","cwd":"{_TEST_CWD}"}}',
                                },
                            }
                        )
                    ],
                ),
                Message(role="assistant", content="done"),
            ]
        )

        class FakeShellExecutor:
            def __init__(self) -> None:
                self.execute = AsyncMock(
                    return_value=ShellExecutionResult(
                        stdout="ok\n",
                        stderr="",
                        exit_code=0,
                        risk_flags=["custom-risk-flag"],
                    )
                )

        orch = AgentOrchestrator(
            llm_client=llm,
            shell_executor=FakeShellExecutor(),
        )
        ctx = make_agent_context(
            config=AgentConfig(
                agent_id="agent",
                model="gpt-4o",
                execution_mode=ExecutionMode.BYPASS,
                shell_tool_config=ShellToolConfig(
                    access_mode=ShellAccessMode.ALL_AGENTS,
                ),
            ),
            allowed_paths=[_TEST_CWD],
        )
        ctx.messages.append(Message(role="user", content="show cwd"))
        _result, report = await orch.run(ctx)
        assert "custom-risk-flag" in report.shell_risk_flags

    @pytest.mark.asyncio
    async def test_shell_tool_events_include_invocation_payload(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(
            side_effect=[
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall.model_validate(
                            {
                                "id": "tc-shell",
                                "function": {
                                    "name": "shell_exec",
                                    "arguments": f'{{"command":"pwd","cwd":"{_TEST_CWD}"}}',
                                },
                            }
                        )
                    ],
                ),
                Message(role="assistant", content="done"),
            ]
        )

        class FakeShellExecutor:
            def __init__(self) -> None:
                self.execute = AsyncMock(
                    return_value=ShellExecutionResult(
                        stdout=f"{_TEST_CWD}\n",
                        stderr="",
                        exit_code=0,
                    )
                )

        bus = EventBus()
        captured: dict[str, list[dict[str, Any]]] = {}
        watched_events = (
            "tool.call.detected",
            "tool.call.start",
            "tool.dispatch.selected",
            "tool.execution.start",
            "tool.execution.end",
            "tool.result.ready",
            "tool.call.end",
        )

        async def _capture(event: Any) -> None:
            captured.setdefault(event.name, []).append(event.payload)

        for event_name in watched_events:
            bus.subscribe(event_name, _capture)

        orch = AgentOrchestrator(
            llm_client=llm,
            shell_executor=FakeShellExecutor(),
            event_bus=bus,
        )
        ctx = make_agent_context(
            config=AgentConfig(
                agent_id="agent",
                model="gpt-4o",
                execution_mode=ExecutionMode.BYPASS,
                shell_tool_config=ShellToolConfig(
                    access_mode=ShellAccessMode.ALL_AGENTS,
                ),
            ),
            allowed_paths=[_TEST_CWD],
        )
        ctx.messages.append(Message(role="user", content="show cwd"))
        await orch.run(ctx)

        for event_name in watched_events:
            assert event_name in captured and captured[event_name]
            payload = captured[event_name][0]
            assert payload["arguments"]["command"] == "pwd"
            assert payload["argument_keys"] == ["command", "cwd"]
            assert json.loads(payload["arguments_json"]) == payload["arguments"]
            assert payload["shell_command"] == "pwd"
            assert (
                payload["shell_cwd"]
                == _TEST_CWD
            )

    @pytest.mark.asyncio
    async def test_registry_tool_events_include_invocation_payload(self) -> None:
        """Non-shell (registry) tool events carry arguments_json and invocation payload."""
        llm = MagicMock()
        llm.complete = AsyncMock(
            side_effect=[
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall.model_validate(
                            {
                                "id": "tc-echo",
                                "function": {
                                    "name": "echo_tool",
                                    "arguments": '{"text":"hello","repeat":2}',
                                },
                            }
                        )
                    ],
                ),
                Message(role="assistant", content="done"),
            ]
        )

        reg = ToolRegistry()
        reg.register(
            ToolDefinition(name="echo_tool", description="Echo text"),
            AsyncMock(
                return_value=ToolResult(
                    tool_call_id="tc-echo",
                    tool_name="echo_tool",
                    content="ok",
                )
            ),
        )

        bus = EventBus()
        captured: dict[str, list[dict[str, Any]]] = {}
        watched_events = (
            "tool.call.detected",
            "tool.call.start",
            "tool.dispatch.selected",
            "tool.execution.start",
            "tool.execution.end",
            "tool.result.ready",
            "tool.call.end",
        )

        async def _capture(event: Any) -> None:
            captured.setdefault(event.name, []).append(event.payload)

        for event_name in watched_events:
            bus.subscribe(event_name, _capture)

        orch = AgentOrchestrator(
            llm_client=llm,
            tool_registry=reg,
            event_bus=bus,
        )
        ctx = make_agent_context(
            config=AgentConfig(
                agent_id="agent",
                model="gpt-4o",
                execution_mode=ExecutionMode.BYPASS,
            ),
        )
        ctx.messages.append(Message(role="user", content="run echo"))
        await orch.run(ctx)

        for event_name in watched_events:
            assert event_name in captured and captured[event_name], event_name
            payload = captured[event_name][0]
            assert "arguments" in payload
            assert payload["arguments"]["text"] == "hello"
            assert payload["arguments"]["repeat"] == 2
            assert payload["argument_keys"] == ["repeat", "text"]
            assert json.loads(payload["arguments_json"]) == payload["arguments"]
            # Non-shell must not have shell_command / shell_cwd
            assert "shell_command" not in payload
            assert "shell_cwd" not in payload

    def test_tool_dispatch_parse_arguments_rejects_non_object_json(self) -> None:
        ctx = make_agent_context(
            config=AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
        report = make_execution_report(context=ctx)
        with pytest.raises(ValueError, match="must decode to a JSON object"):
            ToolDispatcher._parse_tool_arguments(
                tool_name="echo",
                raw_args='["x"]',
                report=report,
            )
        assert "tool_arguments_not_object:echo" in report.warnings

    def test_tool_dispatch_parse_arguments_rejects_invalid_type(self) -> None:
        ctx = make_agent_context(
            config=AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
        report = make_execution_report(context=ctx)
        with pytest.raises(ValueError, match="must be provided as a JSON object or string"):
            ToolDispatcher._parse_tool_arguments(
                tool_name="echo",
                raw_args=123,
                report=report,
            )
        assert "tool_arguments_invalid_type:echo" in report.warnings

    @pytest.mark.asyncio
    async def test_tool_preflight_denies_shell_without_capability(self) -> None:
        hooks = HookManager()
        policy_runner = PolicyRunner(timeout_policy=None, retry_policy=None)
        shell_handler = ShellHandler(
            shell_executor=None,
            policy_runner=policy_runner,
            shell_safety_policy=None,
            append_tool_results_as_messages=lambda _messages, _results: None,
        )
        dispatcher = ToolDispatcher(
            hooks=hooks,
            event_bus=EventBus(),
            policy=None,
            tool_registry=None,
            tool_executor=None,
            shell_executor=None,
            shell_safety_policy=None,
            shell_handler=shell_handler,
            policy_runner=policy_runner,
        )
        ctx = make_agent_context(
            config=AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        )
        report = make_execution_report(context=ctx)
        decision, source = await dispatcher.evaluate_tool_preflight(
            tool_name="shell_exec",
            arguments={"command": "pwd"},
            context=ctx,
            report=report,
            shell_capability=None,
            shell_tool_name="shell_exec",
        )
        assert decision is not None and decision.value == "deny"
        assert source == "shell_capability"

    @pytest.mark.asyncio
    async def test_auto_compact_tokens_accumulate_across_multiple_calls(self) -> None:
        compressor = MagicMock()
        compressor.apply_auto = AsyncMock(
            return_value=(
                [Message(role="assistant", content="summary")],
                CompactionSummary(current_goal="goal"),
                True,
            )
        )
        cfg = AgentConfig(
            agent_id="agent",
            model="gpt-4o",
            execution_mode=ExecutionMode.BYPASS,
            auto_compact_threshold=0,
        )
        ctx = make_agent_context(config=cfg)
        report = make_execution_report(context=ctx)
        messages = [
            Message(role="user", content="first input"),
            Message(role="tool", content="tool output", tool_call_id="tc", name="tool"),
        ]

        orch = AgentOrchestrator(llm_client=_final_llm(), compressor=compressor)
        _, report = await orch._pre_llm_hooks(
            messages, ctx, report, run_kind=RunKind.LEADER
        )
        _, report = await orch._pre_llm_hooks(
            messages, ctx, report, run_kind=RunKind.LEADER
        )

        assert report.auto_compact_applied == 2
        assert report.tokens_before_compression_total is not None and report.tokens_before_compression_total > 0
        assert report.tokens_after_compression_total is not None and report.tokens_after_compression_total > 0

    @pytest.mark.asyncio
    async def test_manual_compact_tokens_accumulate_across_multiple_calls(self) -> None:
        compressor = MagicMock()
        compressor.apply_manual = AsyncMock(
            return_value=(
                [Message(role="assistant", content="summary")],
                CompactionSummary(current_goal="goal"),
            )
        )
        cfg = AgentConfig(agent_id="agent", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages = MessageList([Message(role="user", content="manual compact me")])
        report = make_execution_report(context=ctx)

        orch = AgentOrchestrator(llm_client=_final_llm(), compressor=compressor)
        await orch.trigger_manual_compact(ctx, report)
        await orch.trigger_manual_compact(ctx, report)

        assert report.manual_compact_applied == 2
        assert report.tokens_before_compression_total is not None and report.tokens_before_compression_total > 0
        assert report.tokens_after_compression_total is not None and report.tokens_after_compression_total > 0

    @pytest.mark.asyncio
    async def test_cancel_without_transport_adds_warning(self) -> None:
        llm = MagicMock()

        async def raise_cancel(*args: Any, **kwargs: Any) -> Message:
            raise asyncio.CancelledError("stop")

        llm.complete = AsyncMock(side_effect=raise_cancel)
        cfg = AgentConfig(agent_id="leader", model="gpt-4o", execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="cancel"))
        ctx.metadata["_active_child_agent_ids"] = ["child-a"]

        orch = AgentOrchestrator(llm_client=llm)
        _, report = await orch.run(ctx)

        assert report.status == ExecutionStatus.CANCELLED
        assert "cancel_transport_not_configured" in report.warnings

    def test_make_control_envelope_requires_control_command(self) -> None:
        from protocore.factories import make_control_envelope

        env = make_control_envelope(
            sender_id="leader",
            recipient_id="sub",
            command=ControlCommand.CANCEL,
            trace_id="t",
            session_id="s",
        )
        assert env.payload["command"] == "cancel"

    def test_get_event_bus_is_deprecated_alias(self) -> None:
        from protocore.events import EventBus, create_event_bus, get_event_bus

        bus = create_event_bus()
        assert isinstance(bus, EventBus)
        with pytest.warns(DeprecationWarning):
            legacy_bus = get_event_bus("default")
        assert isinstance(legacy_bus, EventBus)

    def test_extract_task_aggregates_all_user_messages(self) -> None:
        from protocore.orchestrator_utils import extract_task

        messages = [
            Message(role="assistant", content="draft"),
            Message(role="user", content="Build the feature"),
            Message(role="assistant", content="thinking"),
            Message(role="user", content="Also add tests"),
        ]

        assert extract_task(messages) == "Build the feature\n\nAlso add tests"
        assert extract_task(messages, strategy="last") == "Also add tests"
        assert extract_task([Message(role="assistant", content="no task")]) is None

    def test_merge_execution_report_deduplicates_subagent_runs_and_compression_events(self) -> None:
        from protocore.orchestrator_utils import merge_execution_report
        from protocore.types import CompressionEvent, ExecutionReport, SubagentRunSummary

        event = CompressionEvent(
            kind="auto",
            tokens_before=10,
            tokens_after=5,
            timestamp="2026-03-08T00:00:00+00:00",
        )
        duplicate_event = CompressionEvent(
            kind="auto",
            tokens_before=999,
            tokens_after=1,
            timestamp="2026-03-08T00:00:00+00:00",
        )
        subagent_run = SubagentRunSummary(
            agent_id="child-1",
            status=ExecutionStatus.COMPLETED,
            started_at="2026-03-08T00:00:00+00:00",
            finished_at="2026-03-08T00:00:01+00:00",
            duration_ms=1000.0,
        )
        duplicate_subagent_run = SubagentRunSummary(
            agent_id="child-1",
            status=ExecutionStatus.FAILED,
            started_at="2026-03-08T00:00:00+00:00",
            finished_at="2026-03-08T00:00:09+00:00",
            duration_ms=9000.0,
            errors=["duplicate"],
        )
        target = ExecutionReport(
            agent_id="leader",
            compression_events=[event],
            subagent_runs=[subagent_run],
        )
        source = ExecutionReport(
            agent_id="leader",
            status=ExecutionStatus.COMPLETED,
            finished_at="2026-03-08T00:00:02+00:00",
            duration_ms=2000.0,
            compression_events=[duplicate_event],
            subagent_runs=[duplicate_subagent_run],
        )

        merge_execution_report(target, source)

        assert len(target.compression_events) == 1
        assert len(target.subagent_runs) == 1

    def test_execution_report_finalize_logs_repeated_call(self, caplog: pytest.LogCaptureFixture) -> None:
        from protocore.types import ExecutionReport

        report = ExecutionReport(agent_id="agent")

        report.finalize(ExecutionStatus.COMPLETED)
        with caplog.at_level("DEBUG"):
            report.finalize(ExecutionStatus.FAILED)

        assert "ignored repeated call" in caplog.text

    def test_agent_envelope_missing_payload_model_raises_clear_error(self) -> None:
        from pydantic import ValidationError
        from protocore import types as core_types
        from protocore.types import AgentEnvelope, AgentIdentity, AgentRole, MessageType

        payload_model = core_types._PAYLOAD_MODELS.pop(MessageType.RESULT)
        try:
            with pytest.raises(ValidationError, match="No payload model registered"):
                AgentEnvelope(
                    message_type=MessageType.RESULT,
                    sender=AgentIdentity(agent_id="leader", role=AgentRole.LEADER),
                    recipient=AgentIdentity(agent_id="sub", role=AgentRole.SUBAGENT),
                    payload={"status": "success", "summary": "done"},
                )
        finally:
            core_types._PAYLOAD_MODELS[MessageType.RESULT] = payload_model

    @pytest.mark.asyncio
    async def test_auto_select_requires_non_empty_task(self) -> None:
        selection = MagicMock()
        selection.select = AsyncMock(return_value="child-1")
        registry = AgentRegistry()
        registry.register(
            AgentConfig(
                agent_id="child-1",
                model="gpt-4o",
                role=AgentRole.SUBAGENT,
                execution_mode=ExecutionMode.BYPASS,
            )
        )
        ctx = make_agent_context(
            config=AgentConfig(
                agent_id="leader",
                model="gpt-4o",
                role=AgentRole.LEADER,
                execution_mode=ExecutionMode.AUTO_SELECT,
            )
        )
        ctx.messages.append(Message(role="assistant", content="no user task here"))

        orch = AgentOrchestrator(
            llm_client=_final_llm(),
            subagent_selection_policy=selection,
            agent_registry=registry,
        )
        result, report = await orch.run(ctx, run_kind=RunKind.LEADER)

        assert result.status == ExecutionStatus.FAILED
        assert report.error_code == "EMPTY_TASK"
        selection.select.assert_not_awaited()

    def test_append_tool_results_as_messages_marks_error_and_untrusted(self) -> None:
        from protocore.orchestrator_utils import append_tool_results_as_messages

        messages: list[Message] = []
        append_tool_results_as_messages(
            messages,
            [
                ToolResult(
                    tool_call_id="tc1",
                    tool_name="scan",
                    content="suspicious payload",
                    is_error=True,
                    prompt_injection_signal=True,
                    metadata={"severity": "high"},
                )
            ],
        )

        assert len(messages) == 1
        assert messages[0].role == "tool"
        assert messages[0].content == "[TOOL ERROR]\n[UNTRUSTED OUTPUT]\nsuspicious payload"

    @pytest.mark.asyncio
    async def test_auto_compact_timeout_retries_and_falls_back(self) -> None:
        from protocore.compression import auto_compact

        llm = MagicMock()

        async def slow_complete(*args: Any, **kwargs: Any) -> Message:
            await asyncio.sleep(0.05)
            return Message(role="assistant", content='{"current_goal":"never reached"}')

        llm.complete = AsyncMock(side_effect=slow_complete)
        cfg = AgentConfig(
            agent_id="agent",
            model="gpt-4o",
            auto_compact_threshold=0,
            auto_compact_timeout_seconds=0.01,
        )
        messages = [Message(role="user", content="compress me")]

        new_messages, summary, parse_ok = await auto_compact(
            messages,
            llm_client=llm,
            model=cfg.model,
            config=cfg,
        )

        assert parse_ok is False
        assert summary is not None
        assert "[summarization failed; continuing]" in summary.current_goal
        assert new_messages[0].role == "system"
        assert llm.complete.await_count == 2

    def test_thinking_profile_resolution_uses_live_registry(self) -> None:
        from protocore.types import ThinkingProfilePreset

        original = ThinkingProfileRegistry.get("thinking_planner")
        try:
            cfg = AgentConfig(
                agent_id="agent",
                model="gpt-4o",
                thinking_profile=ThinkingProfilePreset.THINKING_PLANNER,
            )
            ThinkingProfileRegistry.register(
                "thinking_planner",
                {"temperature": 0.9, "enable_thinking": False},
            )

            resolved = cfg.resolved_with_selective_thinking()

            assert resolved.temperature == 0.9
            assert resolved.enable_thinking is False
        finally:
            if original is not None:
                ThinkingProfileRegistry.register("thinking_planner", original)

    @pytest.mark.asyncio
    async def test_child_orchestrator_uses_isolated_hook_manager(self) -> None:
        class SessionTracker:
            def __init__(self) -> None:
                self.seen_agent_ids: list[str] = []

            @hookimpl
            def on_session_start(self, context: Any, report: Any) -> None:
                _ = report
                self.seen_agent_ids.append(context.config.agent_id)

        tracker = SessionTracker()
        hooks = HookManager()
        hooks.register(tracker)

        selection = MagicMock()
        selection.select = AsyncMock(return_value="child-1")
        registry = AgentRegistry()
        registry.register(
            AgentConfig(
                agent_id="child-1",
                model="gpt-4o",
                role=AgentRole.SUBAGENT,
                execution_mode=ExecutionMode.LEADER,
            )
        )

        parent = AgentConfig(
            agent_id="leader",
            model="gpt-4o",
            role=AgentRole.LEADER,
            execution_mode=ExecutionMode.AUTO_SELECT,
        )
        ctx = make_agent_context(config=parent)
        ctx.messages.append(Message(role="user", content="delegate this"))

        orch = AgentOrchestrator(
            llm_client=_final_llm(
                '{"status":"success","summary":"done","artifacts":[],"files_changed":[],"tool_calls_made":0,"errors":[],"next_steps":null}'
            ),
            hook_manager=hooks,
            subagent_selection_policy=selection,
            agent_registry=registry,
        )
        result, report = await orch.run(ctx, run_kind=RunKind.LEADER)

        assert result.status == ExecutionStatus.COMPLETED
        assert report.status == ExecutionStatus.COMPLETED
        assert tracker.seen_agent_ids == ["leader"]

    @pytest.mark.asyncio
    async def test_event_bus_handlers_run_sequentially(self) -> None:
        from protocore.events import CoreEvent

        bus = EventBus()
        calls: list[str] = []

        async def handler_a(event: Any) -> None:
            _ = event
            calls.append("a:start")
            await asyncio.sleep(0.01)
            calls.append("a:end")

        async def handler_b(event: Any) -> None:
            _ = event
            calls.append("b")

        bus.subscribe("test.event", handler_a)
        bus.subscribe("test.event", handler_b)
        await bus.emit(CoreEvent(name="test.event", payload={}))

        assert calls == ["a:start", "a:end", "b"]
