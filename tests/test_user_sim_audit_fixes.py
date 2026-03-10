from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from protocore import (
    AgentConfig,
    AgentOrchestrator,
    ApiMode,
    accumulate_usage_from_llm_calls,
    CapabilityBasedSelectionPolicy,
    CompactionSummary,
    CompressionEvent,
    ExecutionMode,
    ExecutionStatus,
    ExecutionReport,
    EventBus,
    HookManager,
    LLMUsage,
    Message,
    OpenAILLMClient,
    Result,
    SubagentResult,
    SubagentStatus,
    ToolDefinition,
    ToolCallRecord,
    ToolContext,
    ToolResult,
    RunKind,
    hookimpl,
    get_text_content,
    make_agent_context,
    resume_from_pending,
)
from protocore.orchestrator_errors import ContractViolationError
from protocore.registry import AgentRegistry
from protocore.testing import FakeLLMClient
from protocore.compression import auto_compact
from protocore.orchestrator_utils import serialize_messages_for_api
from protocore.orchestrator_state import accumulate_llm_usage
from protocore.orchestrator_utils import merge_execution_report
from protocore.types import ToolCall


class _FallbackToCoder:
    async def select(self, task: str, available_agents: list[str], context: Any) -> str:
        _ = (task, available_agents, context)
        return "coder"


class _StructuredPayload(BaseModel):
    answer: int


def test_fix_max_tokens_negative_validation_is_explicit() -> None:
    with pytest.raises(ValidationError) as exc_info:
        AgentConfig(agent_id="a", model="m", max_tokens=-1)
    assert "greater than or equal to 1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_fix_empty_messages_rejected_before_provider_call() -> None:
    llm = MagicMock()
    llm.complete = MagicMock()
    orch = AgentOrchestrator(llm_client=llm)
    context = make_agent_context(config=AgentConfig(agent_id="a", model="m"))

    result, report = await orch.run(context)

    assert result.status == ExecutionStatus.FAILED
    assert report.error_code == "EMPTY_MESSAGES"
    llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_fix_orchestrator_structured_output_path() -> None:
    fake = FakeLLMClient(structured_responses=[_StructuredPayload(answer=42)])
    orch = AgentOrchestrator(llm_client=fake)
    context = make_agent_context(
        config=AgentConfig(
            agent_id="a",
            model="m",
            execution_mode=ExecutionMode.BYPASS,
            response_format=_StructuredPayload,
        )
    )
    context.messages.append(Message(role="user", content="Return JSON"))

    result, report = await orch.run(context)

    assert result.status == ExecutionStatus.COMPLETED
    assert report.status == ExecutionStatus.COMPLETED
    assert result.metadata["structured"]["answer"] == 42
    assert fake.complete_structured_calls


@pytest.mark.asyncio
async def test_fix_structured_output_usage_accumulates_into_report() -> None:
    fake = FakeLLMClient(
        structured_responses=[
            _StructuredPayload(answer=7),
        ]
    )
    payload = fake._structured_responses[0]
    object.__setattr__(payload, "__protocore_usage__", LLMUsage(input_tokens=13, output_tokens=5))
    orch = AgentOrchestrator(llm_client=fake)
    context = make_agent_context(
        config=AgentConfig(
            agent_id="a",
            model="m",
            execution_mode=ExecutionMode.BYPASS,
            response_format=_StructuredPayload,
        )
    )
    context.messages.append(Message(role="user", content="Return JSON"))

    _result, report = await orch.run(context)

    assert report.input_tokens == 13
    assert report.output_tokens == 5


@pytest.mark.asyncio
async def test_fix_auto_select_merges_usage_and_treats_unstructured_success_as_partial() -> None:
    registry = AgentRegistry()
    registry.register(
        AgentConfig(
            agent_id="child",
            model="m",
            execution_mode=ExecutionMode.BYPASS,
        )
    )
    selection = MagicMock()
    selection.select = AsyncMock(return_value="child")
    fake = FakeLLMClient(
        complete_responses=[
            Message(
                role="assistant",
                content="I finished the task successfully.",
                usage=LLMUsage(input_tokens=11, output_tokens=7, cached_tokens=0),
            )
        ]
    )
    orch = AgentOrchestrator(
        llm_client=fake,
        agent_registry=registry,
        subagent_selection_policy=selection,
    )
    context = make_agent_context(
        config=AgentConfig(
            agent_id="leader",
            model="m",
            execution_mode=ExecutionMode.AUTO_SELECT,
        )
    )
    context.messages.append(Message(role="user", content="delegate to child"))

    result, report = await orch.run(context)

    assert result.status == ExecutionStatus.PARTIAL
    assert report.status == ExecutionStatus.PARTIAL
    assert report.input_tokens == 11
    assert report.output_tokens == 7
    assert report.metadata["auto_selected_agent"] == "child"


@pytest.mark.asyncio
async def test_fix_capability_policy_drops_substring_bias() -> None:
    fake = FakeLLMClient(
        complete_responses=[
            Message(
                role="assistant",
                content=(
                    "The request mentions coder and math-subagent capabilities, "
                    "but this text is not strict output."
                ),
            )
        ]
    )
    policy = CapabilityBasedSelectionPolicy(
        llm_client=fake,
        fallback_policy=_FallbackToCoder(),
    )
    context = make_agent_context(config=AgentConfig(agent_id="leader", model="m"))

    selected = await policy.select(
        task="write Python code",
        available_agents=["math-subagent", "coder"],
        context=context,
    )

    assert selected == "coder"


def test_fix_selection_policy_alias_and_runtime_setter() -> None:
    policy_a = MagicMock()
    policy_b = MagicMock()
    with pytest.warns(DeprecationWarning):
        orch = AgentOrchestrator(
            llm_client=FakeLLMClient(default_complete_response=Message(role="assistant", content="ok")),
            selection_policy=policy_a,
        )
    orch.set_subagent_selection_policy(policy_b)
    assert orch._selection_policy is policy_b


def test_fix_openai_client_supports_timeout_ctor() -> None:
    with patch("protocore.integrations.llm.openai_client.AsyncOpenAI") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        _ = OpenAILLMClient(api_key="x", timeout=5.0)
    assert mock_client_cls.call_args is not None
    assert mock_client_cls.call_args.kwargs["timeout"] == 5.0


def test_fix_agent_context_messages_rejects_non_message_append() -> None:
    context = make_agent_context(config=AgentConfig(agent_id="a", model="m"))

    with pytest.raises(TypeError) as exc_info:
        context.messages.append(MagicMock())

    assert "Message instances" in str(exc_info.value)
    assert "Convert Result objects explicitly" in str(exc_info.value)


def test_fix_tool_definition_simple_builder() -> None:
    tool = ToolDefinition.simple(
        name="calculate",
        description="Evaluate expression",
        params={
            "expression": ("string", True, "Expression to evaluate"),
            "precision": ("integer", False, "Optional decimal places"),
        },
    )

    assert tool.parameters.required == ["expression"]
    assert tool.parameters.properties["expression"]["type"] == "string"
    assert tool.parameters.properties["precision"]["description"] == "Optional decimal places"


def test_fix_execution_report_iterations_alias_warns() -> None:
    report = ExecutionReport(
        request_id="r",
        trace_id="t",
        session_id="s",
        agent_id="a",
        run_kind=RunKind.LEADER,
        model="m",
        api_mode=ApiMode.RESPONSES,
        execution_mode=ExecutionMode.BYPASS,
        loop_count=3,
    )

    with pytest.warns(DeprecationWarning):
        assert report.iterations == 3


def test_fix_execution_report_tool_calls_count_alias_warns() -> None:
    report = ExecutionReport(
        request_id="r",
        trace_id="t",
        session_id="s",
        agent_id="a",
        run_kind=RunKind.LEADER,
        model="m",
        api_mode=ApiMode.RESPONSES,
        execution_mode=ExecutionMode.BYPASS,
        tool_calls_total=2,
    )

    with pytest.warns(DeprecationWarning):
        assert report.tool_calls_count == 2


def test_fix_result_helpers_cover_multi_turn_and_structured_output() -> None:
    result = Result(
        content="done",
        metadata={"structured": {"answer": 42}},
    )

    message = result.to_message()
    parsed = result.get_structured(_StructuredPayload)

    assert message.role == "assistant"
    assert message.content == "done"
    assert parsed == _StructuredPayload(answer=42)
    assert Result(content="plain").get_structured(_StructuredPayload) is None


def test_fix_tool_call_record_status_alias() -> None:
    assert ToolCallRecord(tool_name="echo", success=True).status == "success"
    assert ToolCallRecord(tool_name="echo", success=False).status == "failed"


def test_fix_compaction_summary_messages_removed_property() -> None:
    item = CompactionSummary(original_count=10, compacted_count=3)
    assert item.messages_removed == 7


def test_fix_get_text_content_handles_multipart_messages() -> None:
    message = Message.model_validate(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "input_json", "json_data": {"a": 1}},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
        }
    )

    text = get_text_content(message)

    assert "hello" in text
    assert '{"a": 1}' in text
    assert "https://example.com/image.png" in text


def test_fix_serialize_messages_hoists_system_messages_to_front() -> None:
    messages = [
        Message(role="user", content="hello"),
        Message(role="system", content="late system"),
        Message(role="assistant", content="world"),
    ]

    payload = serialize_messages_for_api(
        messages,
        system="top system",
        target_api="chat",
    )

    # Single system message at the beginning, merging param + embedded system
    assert [item["role"] for item in payload] == ["system", "user", "assistant"]
    assert payload[0]["content"] == "top system\n\nlate system"


def test_fix_serialize_messages_single_system_only_at_beginning() -> None:
    """System is always a single message at the very beginning (no system param)."""
    messages = [
        Message(role="system", content="first"),
        Message(role="system", content="second"),
        Message(role="user", content="hi"),
    ]
    payload = serialize_messages_for_api(messages, system=None, target_api="chat")
    system_items = [p for p in payload if p.get("role") == "system"]
    assert len(system_items) == 1
    assert payload[0]["role"] == "system"
    assert payload[0]["content"] == "first\n\nsecond"
    assert payload[1]["role"] == "user"


@pytest.mark.asyncio
async def test_fix_event_bus_deduplicates_specific_and_wildcard_handler() -> None:
    bus = EventBus()
    received: list[str] = []

    async def handler(event: Any) -> None:
        received.append(event.name)

    bus.subscribe("session.start", handler)
    bus.subscribe("*", handler)

    await bus.emit_simple("session.start")

    assert received == ["session.start"]


@pytest.mark.asyncio
async def test_fix_report_metadata_is_synced_from_context_before_return() -> None:
    fake = FakeLLMClient(
        complete_responses=[Message(role="assistant", content="ok")]
    )
    orch = AgentOrchestrator(llm_client=fake)
    context = make_agent_context(
        config=AgentConfig(agent_id="a", model="m", execution_mode=ExecutionMode.AUTO_SELECT),
    )
    registry = AgentRegistry()
    registry.register(AgentConfig(agent_id="writer", model="m", execution_mode=ExecutionMode.BYPASS))
    orch._agent_registry = registry
    selection = MagicMock()
    selection.select = AsyncMock(return_value="writer")
    orch.set_subagent_selection_policy(selection)
    context.messages.append(Message(role="user", content="delegate"))

    _result, report = await orch.run(context)

    assert report.metadata["auto_selected_agent"] == "writer"


@pytest.mark.asyncio
async def test_fix_auto_compact_uses_structured_summary_generation() -> None:
    fake = FakeLLMClient(
        structured_responses=[{"current_goal": "continue"}],
    )

    messages, summary, parse_success = await auto_compact(
        [Message(role="user", content="very long input")],
        llm_client=fake,
        model="m",
        threshold_tokens=0,
    )

    assert parse_success is True
    assert summary is not None
    assert summary.current_goal == "continue"
    assert fake.complete_structured_calls
    assert not fake.complete_calls
    assert messages[0].role == "system"


@pytest.mark.asyncio
async def test_fix_compaction_summary_includes_statistics() -> None:
    fake = FakeLLMClient(
        structured_responses=[{"current_goal": "continue"}],
    )

    _messages, summary, parse_success = await auto_compact(
        [Message(role="user", content="very long input")],
        llm_client=fake,
        model="m",
        threshold_tokens=0,
    )

    assert parse_success is True
    assert summary is not None
    assert summary.original_count == 1
    assert summary.compacted_count >= 1
    assert summary.tokens_saved >= 0
    assert summary.duration_ms is not None


@pytest.mark.asyncio
async def test_fix_error_surface_uses_safe_reason() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=RuntimeError("secret-token=123"))
    orch = AgentOrchestrator(llm_client=llm)
    context = make_agent_context(
        config=AgentConfig(agent_id="a", model="m", execution_mode=ExecutionMode.BYPASS)
    )
    context.messages.append(Message(role="user", content="trigger"))

    result, _report = await orch.run(context)

    assert result.content == "[error: runtimeerror]"
    assert "secret-token=123" not in result.content


@pytest.mark.asyncio
async def test_fix_error_surface_includes_sanitized_upstream_details() -> None:
    class UpstreamError(RuntimeError):
        def __init__(self, message: str, status_code: int) -> None:
            super().__init__(message)
            self.status_code = status_code

    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=UpstreamError("Model 'invalid-model' does not exist", 404)
    )
    orch = AgentOrchestrator(llm_client=llm)
    context = make_agent_context(
        config=AgentConfig(agent_id="a", model="m", execution_mode=ExecutionMode.BYPASS)
    )
    context.messages.append(Message(role="user", content="trigger"))

    result, _report = await orch.run(context)

    assert "upstream 404" in result.content
    assert "invalid-model" in result.content
    assert result.error_details is not None
    assert result.error_details["status_code"] == 404
    assert "invalid-model" in result.error_details["message"]


@pytest.mark.asyncio
async def test_fix_tool_call_details_populated_in_report() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=[
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall.model_validate(
                        {
                            "id": "tc-echo",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"value":"hi"}'},
                        }
                    )
                ],
            ),
            Message(role="assistant", content="done"),
        ]
    )
    orch = AgentOrchestrator(llm_client=llm)
    orch._tool_registry.register(
        ToolDefinition(name="echo", description="echo"),
        AsyncMock(return_value=ToolResult(tool_call_id="tc-echo", tool_name="echo", content="ok")),
    )
    context = make_agent_context(
        config=AgentConfig(
            agent_id="a",
            model="m",
            execution_mode=ExecutionMode.BYPASS,
            tool_definitions=[ToolDefinition(name="echo", description="echo")],
        )
    )
    context.messages.append(Message(role="user", content="run echo"))

    _result, report = await orch.run(context)

    assert report.tool_calls_total == 1
    assert len(report.tool_call_details) == 1
    assert report.tool_call_details[0].tool_name == "echo"
    assert report.tool_call_details[0].arguments == {"value": "hi"}
    assert report.tool_call_details[0].latency_ms is not None
    assert report.tool_call_details[0].success is True


@pytest.mark.asyncio
async def test_fix_hook_manager_preserves_shared_events_for_subagents() -> None:
    class EventsPlugin:
        def __init__(self) -> None:
            self.events: list[str] = []

        @hookimpl
        def on_session_start(self, context: Any, report: Any) -> None:
            _ = report
            self.events.append(context.config.agent_id)

    plugin = EventsPlugin()
    hooks = HookManager()
    hooks.register(plugin)
    selection = MagicMock()
    selection.select = AsyncMock(return_value="child")
    registry = AgentRegistry()
    registry.register(
        AgentConfig(
            agent_id="child",
            model="m",
            execution_mode=ExecutionMode.LEADER,
        )
    )
    context = make_agent_context(
        config=AgentConfig(agent_id="leader", model="m", execution_mode=ExecutionMode.AUTO_SELECT)
    )
    context.messages.append(Message(role="user", content="delegate"))

    orch = AgentOrchestrator(
        llm_client=FakeLLMClient(
            complete_responses=[
                Message(
                    role="assistant",
                    content='{"status":"success","summary":"done","artifacts":[],"files_changed":[],"tool_calls_made":0,"errors":[],"next_steps":null}',
                )
            ]
        ),
        hook_manager=hooks,
        subagent_selection_policy=selection,
        agent_registry=registry,
    )

    _result, report = await orch.run(context)

    assert report.status == ExecutionStatus.COMPLETED
    assert plugin.events == ["leader", "child"]


@pytest.mark.asyncio
async def test_fix_openai_client_stream_alias_delegates_to_stream_with_tools() -> None:
    with patch("protocore.integrations.llm.openai_client.AsyncOpenAI") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        client = OpenAILLMClient(api_key="x", default_model="m")

    fake = FakeLLMClient(default_stream_sequence=[{"type": "done", "usage": {}}])
    spy = MagicMock(
        return_value=fake.stream_with_tools(messages=[Message(role="user", content="hello")])
    )
    client.stream_with_tools = spy  # type: ignore[method-assign]

    events = [event async for event in client.stream(messages=[Message(role="user", content="hello")])]

    assert events == [{"type": "done", "usage": {}}]
    assert spy.call_args.kwargs["tools"] is None


@pytest.mark.asyncio
async def test_fix_stream_with_tools_accepts_tool_registry_kwarg_without_forwarding() -> None:
    with patch("protocore.integrations.llm.openai_client.AsyncOpenAI") as mock_client_cls:
        mock_client = MagicMock()
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = []
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        mock_client_cls.return_value = mock_client
        client = OpenAILLMClient(api_key="x", api_mode=ApiMode.CHAT_COMPLETIONS, default_model="m")

    events = [
        event async for event in client.stream_with_tools(
            messages=[Message(role="user", content="hello")],
            tool_registry=object(),
        )
    ]

    assert events[-1]["type"] == "done"
    assert "tool_registry" not in mock_client.chat.completions.create.call_args.kwargs


@pytest.mark.asyncio
async def test_fix_stream_with_tools_drops_non_provider_kwargs() -> None:
    with patch("protocore.integrations.llm.openai_client.AsyncOpenAI") as mock_client_cls:
        mock_client = MagicMock()
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = []
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        mock_client_cls.return_value = mock_client
        client = OpenAILLMClient(api_key="x", api_mode=ApiMode.CHAT_COMPLETIONS, default_model="m")

    _ = [
        event async for event in client.stream_with_tools(
            messages=[Message(role="user", content="hello")],
            tool_definitions=["unexpected"],
            tool_registry=object(),
            extra_body={"trace": "ok"},
        )
    ]

    forwarded = mock_client.chat.completions.create.call_args.kwargs
    assert "tool_registry" not in forwarded
    assert "tool_definitions" not in forwarded
    assert forwarded["extra_body"] == {"trace": "ok"}


def test_fix_tool_context_manual_test_helper() -> None:
    ctx = ToolContext.for_manual_tests(
        agent_id="shell",
        allowed_paths=["/workspace"],
        tool_call_id="tc-1",
    )

    assert ctx.agent_id == "shell"
    assert ctx.allowed_paths == ["/workspace"]
    assert ctx.tool_call_id == "tc-1"


def test_fix_resume_from_pending_helper_builds_metadata_patch() -> None:
    pending_result = Result(
        content="[approval required]",
        status=ExecutionStatus.PARTIAL,
        metadata={
            "pending_shell_approval": {
                "plan_id": "plan-1",
                "tool_call_id": "tc-1",
                "tool_name": "shell_exec",
                "command": "rm tmp.txt",
                "cwd": "/workspace",
                "timeout_ms": 1000,
                "env": {},
                "reason": "cleanup",
                "metadata": {},
            }
        },
    )

    patch_payload = resume_from_pending(pending_result, "approve")

    assert patch_payload["pending_shell_approval"]["plan_id"] == "plan-1"
    assert patch_payload["shell_approval_decisions"] == {"plan-1": "approve"}


@pytest.mark.asyncio
async def test_fix_resume_requires_pending_plan_for_shell_decisions() -> None:
    fake = FakeLLMClient(default_complete_response=Message(role="assistant", content="ok"))
    orch = AgentOrchestrator(llm_client=fake)
    context = make_agent_context(
        config=AgentConfig(agent_id="a", model="m", execution_mode=ExecutionMode.BYPASS)
    )
    context.metadata["shell_approval_decisions"] = {"plan-1": "approve"}
    context.messages.append(Message(role="user", content="hello"))

    result, report = await orch.run(context)

    assert result.status == ExecutionStatus.FAILED
    assert report.error_code == "PENDING_SHELL_APPROVAL_REQUIRED"
    assert "resume_from_pending" in (report.error_message or "")


def test_fix_parallel_token_accounting_helpers_are_explicit() -> None:
    report = ExecutionReport(
        input_tokens=15,
        output_tokens=9,
        subagent_runs=[],
        metadata={
            "parent_tokens_include_subagents": True,
            "child_tokens_sum_input": 10,
            "child_tokens_sum_output": 4,
        },
    )

    assert report.parent_tokens() == (5, 5)
    assert report.child_tokens_sum() == (10, 4)
    assert report.total_tokens_including_subagents() == (15, 9)


def test_fix_workflow_usage_helper_aggregates_messages_and_dicts() -> None:
    totals = accumulate_usage_from_llm_calls(
        [
            Message(role="assistant", content="a", usage=LLMUsage(input_tokens=3, output_tokens=2)),
            {"input_tokens": 4, "output_tokens": 1, "cached_tokens": 1},
            None,
        ]
    )

    assert totals == (7, 3, 1, 0)


def test_fix_orchestrator_rejects_raw_plugin_manager_for_hook_manager() -> None:
    with pytest.raises(ContractViolationError) as exc_info:
        AgentOrchestrator(
            llm_client=FakeLLMClient(default_complete_response=Message(role="assistant", content="ok")),
            hook_manager=MagicMock(),
        )
    assert "HookManager" in str(exc_info.value)


def test_subagent_prompt_instructions_include_schema() -> None:
    instructions = SubagentResult.prompt_instructions()
    assert "Return ONLY valid JSON" in instructions
    assert "status" in instructions
    assert SubagentStatus.SUCCESS.value in instructions


def test_merge_execution_report_skips_terminal_when_disabled() -> None:
    ctx = make_agent_context(config=AgentConfig(agent_id="a", model="m"))
    target = ExecutionReport(
        request_id=ctx.request_id,
        trace_id=ctx.trace_id,
        session_id=ctx.session_id,
        agent_id=ctx.config.agent_id,
        parent_agent_id=None,
        run_kind=RunKind.LEADER,
        model=ctx.config.model,
        api_mode=ctx.config.api_mode,
        execution_mode=ctx.config.execution_mode,
    )
    source = ExecutionReport(
        request_id="r2",
        trace_id=ctx.trace_id,
        session_id=ctx.session_id,
        agent_id=ctx.config.agent_id,
        parent_agent_id=None,
        run_kind=RunKind.LEADER,
        model=ctx.config.model,
        api_mode=ctx.config.api_mode,
        execution_mode=ctx.config.execution_mode,
    )
    source.status = ExecutionStatus.COMPLETED
    source.finished_at = "2026-03-09T00:00:00+00:00"
    source.duration_ms = 12.0
    source.queue_wait_ms = 5.0
    source.input_tokens = 3
    source.output_tokens = 4
    source.estimated_cost = 1.5
    source.tool_calls_by_name["ping"] = 2
    source.tokens_before_compression_total = 10
    source.tokens_after_compression_total = 6
    source.compression_events.append(
        CompressionEvent(
            kind="auto",
            tokens_before=10,
            tokens_after=6,
            messages_affected=1,
            summary_parse_success=True,
        )
    )

    merge_execution_report(target, source, include_terminal=False)

    assert target.status == ExecutionStatus.RUNNING
    assert target.finished_at is None
    assert target.queue_wait_ms == 5.0
    assert target.input_tokens == 3
    assert target.output_tokens == 4
    assert target.estimated_cost == 1.5
    assert target.tool_calls_by_name["ping"] == 2
    assert target.tokens_before_compression_total == 10
    assert target.tokens_after_compression_total == 6
    assert len(target.compression_events) == 1


def test_accumulate_llm_usage_counts_cached_cost() -> None:
    ctx = make_agent_context(
        config=AgentConfig(
            agent_id="a",
            model="m",
            cost_per_token=1.0,
            cost_per_cached_token=0.1,
        )
    )
    report = ExecutionReport(
        request_id=ctx.request_id,
        trace_id=ctx.trace_id,
        session_id=ctx.session_id,
        agent_id=ctx.config.agent_id,
        parent_agent_id=None,
        run_kind=RunKind.LEADER,
        model=ctx.config.model,
        api_mode=ctx.config.api_mode,
        execution_mode=ctx.config.execution_mode,
    )

    accumulate_llm_usage(
        report=report,
        context=ctx,
        input_tokens=10,
        output_tokens=2,
        cached_tokens=4,
    )

    assert report.estimated_cost == 8.4
