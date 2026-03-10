from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from protocore import (
    AgentConfig,
    ExecutionMode,
    ExecutionStatus,
    Message,
    PlanArtifact,
    PolicyDecision,
    Result,
    RunKind,
    StopReason,
    SubagentRunSummary,
    ToolDefinition,
    make_agent_context,
    make_execution_report,
)
from protocore.types import ToolCall
from protocore.orchestrator_utils import (
    PolicyRunner,
    _serialize_function_output,
    _summarize_json_payload,
    build_llm_request_kwargs,
    ensure_terminal_report,
    load_existing_plan,
    merge_execution_report,
    normalize_policy_decision,
    recover_tool_calls_from_assistant_text,
    serialize_messages_for_api,
)


class _RetryOncePolicy:
    def should_retry(self, attempt: int, error: Exception) -> bool:
        _ = error
        return attempt < 2

    def delay_seconds(self, attempt: int) -> float:
        _ = attempt
        return 0.01


class _TinyTimeoutPolicy:
    def get_timeout(self, operation: str) -> float:
        _ = operation
        return 0.001


class TestOrchestratorUtilsSerialization:
    def test_serialize_function_output_supports_content_parts_and_fallbacks(self) -> None:
        parts = [
            SimpleNamespace(type="text", text="alpha"),
            SimpleNamespace(type="input_json", json_data={"k": "v"}),
            SimpleNamespace(type="image_url", image_url={"url": "https://example.com/a.png"}),
        ]
        assert _serialize_function_output(None) == ""
        assert _serialize_function_output("raw") == "raw"
        assert _serialize_function_output(123) == "123"
        assert _serialize_function_output(parts) == 'alpha\n{"k": "v"}\nhttps://example.com/a.png'

    def test_serialize_messages_for_responses_keeps_content_and_function_calls(self) -> None:
        tool_call = ToolCall.model_validate(
            {"id": "call-1", "function": {"name": "search", "arguments": '{"q":"docs"}'}}
        )
        messages = [
            Message(role="assistant", content="thinking", tool_calls=[tool_call]),
            Message(role="tool", content="result", tool_call_id="call-1", name="search"),
        ]

        items = serialize_messages_for_api(messages, system=None, target_api="responses")

        assert items[0]["role"] == "assistant"
        assert items[1]["type"] == "function_call"
        assert items[1]["name"] == "search"
        assert items[2] == {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "result",
        }

    def test_build_llm_request_kwargs_merges_extra_body_and_thinking_and_tools(self) -> None:
        cfg = AgentConfig(
            agent_id="agent",
            model="gpt-4o",
            execution_mode=ExecutionMode.BYPASS,
            top_p=0.9,
            presence_penalty=0.1,
            top_k=20,
            enable_thinking=True,
            parallel_tool_calls=True,
            llm_extra_body={"x": 1},
            llm_request_kwargs={"extra_body": {"chat_template_kwargs": {"preset": "p"}}},
        )

        kwargs = build_llm_request_kwargs(cfg, has_tools=True)

        assert kwargs["top_p"] == 0.9
        assert kwargs["presence_penalty"] == 0.1
        assert kwargs["parallel_tool_calls"] is True
        assert kwargs["extra_body"]["x"] == 1
        assert kwargs["extra_body"]["top_k"] == 20
        assert kwargs["extra_body"]["chat_template_kwargs"]["preset"] == "p"
        assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


class TestOrchestratorUtilsCoreBehaviors:
    def test_normalize_policy_decision_handles_nested_collections_and_invalid_values(self) -> None:
        assert normalize_policy_decision(None) is None
        assert normalize_policy_decision(["invalid", ("allow",)]) == PolicyDecision.ALLOW
        assert normalize_policy_decision("  deny ") == PolicyDecision.DENY
        assert normalize_policy_decision({"not": "supported"}) is None

    def test_load_existing_plan_accepts_model_and_dict(self) -> None:
        cfg = AgentConfig(agent_id="a", model="gpt-4o")
        context = make_agent_context(config=cfg)

        plan = PlanArtifact(steps=[], raw_plan="x")
        context.metadata["plan_artifact"] = plan
        assert load_existing_plan(context) == plan

        context.metadata["plan_artifact"] = plan.model_dump(mode="json")
        loaded = load_existing_plan(context)
        assert isinstance(loaded, PlanArtifact)
        assert loaded.plan_id == plan.plan_id

        context.metadata["plan_artifact"] = "invalid"
        assert load_existing_plan(context) is None

    def test_ensure_terminal_report_normalizes_running_and_adds_error_message(self) -> None:
        cfg = AgentConfig(agent_id="a", model="gpt-4o")
        context = make_agent_context(config=cfg)
        report = make_execution_report(context=context)

        running_result = Result(content="still running", status=ExecutionStatus.RUNNING)
        ensure_terminal_report(report, running_result)

        assert report.status == ExecutionStatus.COMPLETED
        assert report.stop_reason == StopReason.END_TURN

        report2 = make_execution_report(context=context)
        partial_result = Result(content="tool failed details", status=ExecutionStatus.PARTIAL)
        ensure_terminal_report(report2, partial_result)
        assert report2.error_message == "tool failed details"

    def test_merge_execution_report_deduplicates_lists_and_keeps_union(self) -> None:
        cfg = AgentConfig(agent_id="a", model="gpt-4o")
        ctx = make_agent_context(config=cfg)
        target = make_execution_report(context=ctx)
        source = make_execution_report(context=ctx)

        now = datetime.now(timezone.utc).isoformat()
        run = SubagentRunSummary(
            agent_id="child",
            run_kind=RunKind.SUBAGENT,
            status=ExecutionStatus.COMPLETED,
            started_at=now,
            finished_at=now,
            duration_ms=1.0,
        )

        target.shell_risk_flags = ["high"]
        source.shell_risk_flags = ["high", "critical"]
        target.artifacts = ["a.txt"]
        source.artifacts = ["a.txt", "b.txt"]
        target.files_changed = ["f1"]
        source.files_changed = ["f1", "f2"]
        target.warnings = ["w1"]
        source.warnings = ["w1", "w2"]
        target.subagent_runs = [run]
        source.subagent_runs = [run]

        merge_execution_report(target, source)

        assert target.shell_risk_flags == ["high", "critical"]
        assert target.artifacts == ["a.txt", "b.txt"]
        assert target.files_changed == ["f1", "f2"]
        assert target.warnings == ["w1", "w2"]
        assert len(target.subagent_runs) == 1

    def test_recover_tool_calls_from_assistant_text_supports_json_and_legacy_tags(self) -> None:
        tools = [ToolDefinition(name="search", description="Search"), ToolDefinition(name="echo", description="Echo")]
        content = """
        [{"name":"search","arguments":{"q":"docs"}}]
        <function=echo><parameter=text>hi</parameter></function>
        """

        recovered = recover_tool_calls_from_assistant_text(content, tools, blocked_tool_names={"search"})

        assert len(recovered) == 1
        assert recovered[0]["function"]["name"] == "echo"

    def test_summarize_json_payload_fallback_for_non_serializable_objects(self) -> None:
        class NotSerializable:
            pass

        summarized = _summarize_json_payload({"obj": NotSerializable()})
        assert summarized == {"type": "dict"}


class TestPolicyRunnerRetries:
    @pytest.mark.asyncio
    async def test_policy_runner_retries_on_timeout_and_sleeps_before_retry(self) -> None:
        cfg = AgentConfig(agent_id="a", model="gpt-4o")
        context = make_agent_context(config=cfg)
        report = make_execution_report(context=context)

        runner = PolicyRunner(
            timeout_policy=_TinyTimeoutPolicy(),
            retry_policy=_RetryOncePolicy(),
        )
        attempts = {"count": 0}

        async def flaky_call() -> str:
            attempts["count"] += 1
            if attempts["count"] == 1:
                await asyncio.sleep(0.02)
            return "ok"

        result = await runner.call(operation="op", report=report, fn=flaky_call)

        assert result == "ok"
        assert attempts["count"] == 2
        assert any("retry:op:attempt=1:TimeoutError" in warning for warning in report.warnings)
