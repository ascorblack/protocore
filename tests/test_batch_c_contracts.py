from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from protocore import (
    AgentConfig,
    AgentContext,
    AgentEnvelope,
    AgentIdentity,
    AgentOrchestrator,
    AgentRole,
    ApiMode,
    CancellationContext,
    CompactionSummary,
    ContentPart,
    ExecutionMode,
    ExecutionStatus,
    Message,
    MessageType,
    RunKind,
    ShellCommandPlan,
    SubagentResult,
    ThinkingProfilePreset,
    ToolResult,
)
from protocore.constants import COMPACTION_SUMMARY_MARKER, THINKING_PROFILE_DEFAULTS, ThinkingProfileRegistry
from protocore.context import estimate_llm_prompt_tokens
from protocore.ingress import parse_envelope as parse_envelope_impl
from protocore.integrations.llm.openai_client import (
    OpenAILLMClient,
    _parse_chat_output,
    _tools_to_chat_format,
    _tools_to_responses_format,
)
from protocore.json_utils import structured_json_candidates
from protocore.orchestrator_utils import (
    recover_tool_calls_from_assistant_text,
    serialize_messages_for_api,
    resolve_effective_llm_config,
    tool_payload_summary,
)
from protocore.protocols import CompressionStrategy
from protocore.types import (
    ExecutionReport,
    MessageList,
    SessionSnapshot,
    ToolCall,
    ToolCallFunction,
    ToolContext,
    ToolDefinition,
)


def _ctx(**config_kwargs: Any) -> AgentContext:
    cfg = AgentConfig(
        agent_id=config_kwargs.pop("agent_id", "agent"),
        model=config_kwargs.pop("model", "gpt-4o"),
        execution_mode=config_kwargs.pop("execution_mode", ExecutionMode.BYPASS),
        **config_kwargs,
    )
    return AgentContext(config=cfg)


class TestBatchCM019M020:
    @pytest.mark.asyncio
    async def test_prompt_estimation_counts_system_prompt_and_input_json(self) -> None:
        messages = [
            Message(
                role="user",
                content=[
                    ContentPart(type="text", text="hello"),
                    ContentPart(type="input_json", json_data={"payload": "x" * 200}),
                ],
            )
        ]
        small = estimate_llm_prompt_tokens(messages, system="short", api_mode=ApiMode.RESPONSES)
        large = estimate_llm_prompt_tokens(
            messages,
            system="large-system-" + ("y" * 500),
            api_mode=ApiMode.RESPONSES,
        )
        assert large > small

    def test_structured_json_candidates_support_inline_json_fence_without_newline(self) -> None:
        candidates = structured_json_candidates("```json{\"ok\":true}```")

        assert '{"ok":true}' in candidates

    def test_serialize_messages_for_api_covers_multimodal_and_tool_output_shapes(self) -> None:
        class UnknownPart:
            type = "audio_url"

        class ToolMessage:
            role = "tool"
            content = 123
            tool_calls = None
            tool_call_id = "call-1"
            name = "echo"

        class NumericUserMessage:
            role = "user"
            content = 99
            tool_calls = None
            tool_call_id = None
            name = None

        class RichToolMessage:
            role = "tool"
            content = [
                ContentPart(type="text", text="alpha"),
                ContentPart(type="input_json", json_data={"k": "v"}),
                ContentPart(type="image_url", image_url={"url": "https://example.com/b.png"}),
            ]
            tool_calls = None
            tool_call_id = "call-2"
            name = "echo"

        messages = [
            Message(
                role="user",
                content=[
                    ContentPart(type="text", text="hello"),
                    ContentPart(
                        type="image_url",
                        image_url={"url": "https://example.com/a.png", "detail": "high"},
                    ),
                    ContentPart(type="input_json", json_data={"k": "v"}),
                ],
            ),
        ]
        unknown_content_message = type(
            "UnknownContentMessage",
            (),
            {
                "role": "user",
                "content": [UnknownPart()],
                "tool_calls": None,
                "tool_call_id": None,
                "name": None,
            },
        )()

        chat_messages: list[Any] = messages + [unknown_content_message, NumericUserMessage()]
        chat_payload = serialize_messages_for_api(
            chat_messages,
            system=None,
            target_api="chat",
        )
        responses_messages: list[Any] = messages + [
            unknown_content_message,
            ToolMessage(),
            RichToolMessage(),
        ]
        responses_payload = serialize_messages_for_api(
            responses_messages,
            system="system",
            target_api="responses",
        )

        assert chat_payload[0]["content"] == [
            {"type": "text", "text": "hello"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/a.png", "detail": "high"},
            },
            {"type": "text", "text": '{"k": "v"}'},
        ]
        assert chat_payload[1]["content"] == [
            {"type": "audio_url"},
        ]
        assert chat_payload[2]["content"] == "99"
        assert responses_payload[0] == {
            "role": "system",
            "content": [{"type": "input_text", "text": "system"}],
        }
        assert responses_payload[1]["content"] == [
            {"type": "input_text", "text": "hello"},
            {
                "type": "input_image",
                "image_url": "https://example.com/a.png",
                "detail": "high",
            },
            {"type": "input_text", "text": '{"k": "v"}'},
        ]
        assert responses_payload[2]["content"] == [
            {"type": "audio_url"},
        ]
        assert responses_payload[3] == {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "123",
        }
        assert responses_payload[4] == {
            "type": "function_call_output",
            "call_id": "call-2",
            "output": 'alpha\n{"k": "v"}\nhttps://example.com/b.png',
        }

    @pytest.mark.asyncio
    async def test_auto_compact_transcript_keeps_recent_tail_context(self) -> None:
        from protocore.compression import auto_compact

        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value=Message(
                role="assistant",
                content=CompactionSummary(current_goal="done").model_dump_json(),
            )
        )
        messages = [Message(role="user", content=f"old-{idx}") for idx in range(80)]
        messages.append(Message(role="user", content="latest-important-requirement"))

        cfg = AgentConfig(
            agent_id="agent",
            model="gpt-4o",
            execution_mode=ExecutionMode.BYPASS,
            auto_compact_threshold=0,
            system_prompt="system prompt",
        )
        await auto_compact(messages, llm_client=llm, model=cfg.model, config=cfg)

        assert llm.complete.await_args is not None
        transcript_prompt = llm.complete.await_args.kwargs["messages"][0].content
        assert "latest-important-requirement" in str(transcript_prompt)


class TestBatchCM021M025:
    @pytest.mark.asyncio
    async def test_session_snapshot_update_preserves_created_at(self) -> None:
        existing = SessionSnapshot(
            session_id="session",
            trace_id="trace",
            agent_id="agent",
            message_history_ref="session:session:messages",
            execution_metadata_ref="request:req:metadata",
            created_at="2026-03-08T00:00:00+00:00",
        )
        state_manager = MagicMock()
        state_manager.load_session_snapshot = AsyncMock(return_value=existing)
        state_manager.update_session_snapshot = AsyncMock()

        ctx = AgentContext(
            session_id="session",
            trace_id="trace",
            request_id="req",
            config=AgentConfig(agent_id="agent", model="gpt-4o"),
            messages=MessageList([Message(role="user", content="hello")]),
        )
        report = ExecutionReport(
            request_id="req",
            trace_id="trace",
            session_id="session",
            agent_id="agent",
            status=ExecutionStatus.COMPLETED,
            finished_at="2026-03-08T00:00:01+00:00",
            duration_ms=1000.0,
        )
        orch = AgentOrchestrator(llm_client=MagicMock(), state_manager=state_manager)
        await orch._update_session_snapshot(context=ctx, report=report)

        assert state_manager.update_session_snapshot.await_args is not None
        saved = state_manager.update_session_snapshot.await_args.args[0]
        assert saved.created_at == "2026-03-08T00:00:00+00:00"

    def test_direct_agent_context_normalizes_tool_context_identity(self) -> None:
        ctx = AgentContext(
            session_id="session-a",
            trace_id="trace-a",
            request_id="request-a",
            config=AgentConfig(agent_id="agent-a", model="gpt-4o"),
        )
        assert ctx.tool_context.session_id == "session-a"
        assert ctx.tool_context.trace_id == "trace-a"
        assert ctx.tool_context.agent_id == "agent-a"

    def test_agent_context_does_not_mutate_passed_tool_context(self) -> None:
        shared_tool_context = ToolContext(
            session_id="external-session",
            trace_id="external-trace",
            agent_id="external-agent",
            metadata={"message_history_ref": "external"},
        )

        ctx = AgentContext(
            session_id="session-a",
            trace_id="trace-a",
            request_id="request-a",
            config=AgentConfig(agent_id="agent-a", model="gpt-4o"),
            tool_context=shared_tool_context,
        )

        assert ctx.tool_context.session_id == "session-a"
        assert shared_tool_context.session_id == "external-session"
        assert shared_tool_context.metadata["message_history_ref"] == "external"

    def test_tool_call_function_allows_vendor_specific_extra_fields(self) -> None:
        function = ToolCallFunction.model_validate(
            {
                "name": "echo",
                "arguments": "{}",
                "vendor_extension": {"provider": "vllm"},
            }
        )

        assert function.name == "echo"
        assert function.model_extra == {"vendor_extension": {"provider": "vllm"}}

    @pytest.mark.asyncio
    async def test_parallel_agent_ids_invalid_shape_surfaces_contract_error(self) -> None:
        policy = MagicMock(
            max_concurrency=1,
            timeout_seconds=1.0,
            cancellation_mode="graceful",
            merge_results=AsyncMock(),
        )
        orch = AgentOrchestrator(llm_client=MagicMock(), parallel_execution_policy=policy)
        ctx = _ctx(execution_mode=ExecutionMode.PARALLEL)
        ctx.messages.append(Message(role="user", content="parallel"))
        ctx.metadata["parallel_agent_ids"] = ["a", {"bad": True}]

        _result, report = await orch.run(ctx)
        assert report.status == ExecutionStatus.FAILED
        assert report.error_code == "PARALLEL_AGENT_IDS_INVALID"


class TestBatchCM023M029:
    @pytest.mark.asyncio
    async def test_runtime_compression_protocol_is_injectable(self) -> None:
        class FakeCompression:
            async def apply_auto(
                self,
                messages: list[Message],
                config: Any,
                *,
                precomputed_tokens: int | None = None,
                run_kind: Any = None,
            ) -> tuple[list[Message], CompactionSummary, bool]:
                _ = (config, precomputed_tokens, run_kind)
                return [Message(role="system", content="summary")], CompactionSummary(current_goal="ok"), True

            async def apply_manual(
                self,
                messages: list[Message],
                *,
                model: str | None = None,
                config: Any = None,
                run_kind: Any = None,
            ) -> tuple[list[Message], CompactionSummary]:
                _ = (messages, model, config, run_kind)
                return [Message(role="system", content="summary")], CompactionSummary(current_goal="ok")

        compressor = FakeCompression()
        assert isinstance(compressor, CompressionStrategy)
        orch = AgentOrchestrator(llm_client=MagicMock(), compressor=compressor)
        ctx = _ctx(auto_compact_threshold=0)
        report = ExecutionReport(agent_id="agent")
        messages = [Message(role="user", content="compress me")]
        new_messages, _report = await orch._pre_llm_hooks(
            messages,
            ctx,
            report,
            run_kind=RunKind.LEADER,
        )
        assert new_messages[0].content == "summary"

    def test_subagent_result_recovers_fenced_json(self) -> None:
        raw = """```json
{"status":"success","summary":"done","artifacts":[],"files_changed":[],"tool_calls_made":0,"errors":[],"next_steps":null}
```"""
        result = SubagentResult.parse_with_fallback(raw, agent_id="child")
        assert result.status.value == "success"
        assert result.summary == "done"

    def test_subagent_result_recovers_json_embedded_in_markdown(self) -> None:
        raw = """Result:

```json
{"status":"success","summary":"done","artifacts":[],"files_changed":[],"tool_calls_made":0,"errors":[],"next_steps":null}
```
"""
        result = SubagentResult.parse_with_fallback(raw, agent_id="child")
        assert result.status.value == "success"
        assert result.summary == "done"

    def test_parse_envelope_revalidates_mutated_existing_envelope(self) -> None:
        envelope = AgentEnvelope(
            message_type=MessageType.TASK,
            sender=AgentIdentity(agent_id="leader", role=AgentRole.LEADER),
            recipient=AgentIdentity(agent_id="sub", role=AgentRole.SUBAGENT),
            payload={"task": "ok"},
        )
        envelope.payload = {"task": object()}
        with pytest.raises(ValidationError):
            parse_envelope_impl(envelope)

    def test_envelope_payload_requires_json_serializable_value(self) -> None:
        with pytest.raises(ValidationError, match="JSON-serializable"):
            AgentEnvelope(
                message_type=MessageType.TASK,
                sender=AgentIdentity(agent_id="leader", role=AgentRole.LEADER),
                recipient=AgentIdentity(agent_id="sub", role=AgentRole.SUBAGENT),
                payload={"task": object()},
            )


class TestBatchCM026M038:
    def test_tool_payload_summary_caps_large_content_json(self) -> None:
        payload = {"items": [{"value": "x" * 200} for _ in range(20)], "token": "secret"}
        summary = tool_payload_summary(
            ToolResult(
                tool_call_id="tc1",
                tool_name="inspect",
                content=json.dumps(payload),
                metadata={},
            )
        )
        assert isinstance(summary["content_json"], dict)
        assert summary["content_json"]["type"] == "dict"
        assert "preview" in summary["content_json"]

    def test_recover_tool_calls_collects_multiple_json_candidates(self) -> None:
        tools = [ToolDefinition(name="search", description="search"), ToolDefinition(name="read", description="read")]
        text = """
before
```json
[{"name":"search","arguments":{"q":"alpha"}},{"name":"read","arguments":{"path":"notes.txt"}}]
```
after {"name":"search","arguments":{"q":"alpha"}}
"""
        recovered = recover_tool_calls_from_assistant_text(text, tools)
        assert [call["function"]["name"] for call in recovered] == ["search", "read"]

    def test_parse_chat_output_rejects_empty_choices_with_typed_error(self) -> None:
        response = MagicMock()
        response.choices = []
        with pytest.raises(ValueError, match="chat_response_missing_choices"):
            _parse_chat_output(response)

    @pytest.mark.asyncio
    async def test_stream_with_tools_uses_stable_event_schema_in_both_paths(self) -> None:
        client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)

        response_completed = MagicMock()
        response_completed.type = "response.completed"
        response_completed.response = MagicMock()
        response_completed.response.output = []
        response_completed.response.output_text = "done"
        response_completed.response.usage = None

        async def fake_responses_stream() -> Any:
            delta = MagicMock()
            delta.type = "response.output_text.delta"
            delta.delta = "hi"
            yield delta
            yield response_completed

        with patch.object(client._client.responses, "create", new_callable=AsyncMock) as create:
            create.return_value = fake_responses_stream()
            events = [
                event
                async for event in client.stream_with_tools(
                    messages=[Message(role="user", content="hello")],
                    model="gpt-4o",
                )
            ]
        assert [event["type"] for event in events] == ["delta", "done"]

        chat_client = OpenAILLMClient(api_key="test", api_mode=ApiMode.CHAT_COMPLETIONS)

        async def fake_chat_stream() -> Any:
            chunk = MagicMock()
            chunk.usage = None
            chunk.choices = [MagicMock(delta=MagicMock(content="chat-hi", tool_calls=[]))]
            yield chunk

        with patch.object(
            chat_client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as create_chat:
            create_chat.return_value = fake_chat_stream()
            chat_events = [
                event
                async for event in chat_client.stream_with_tools(
                    messages=[Message(role="user", content="hello")],
                    model="gpt-4o",
                )
            ]
        assert [event["type"] for event in chat_events] == ["delta", "done"]


class TestBatchCM030M036:
    def test_tool_format_strict_is_aligned_between_chat_and_responses(self) -> None:
        tool = ToolDefinition(name="echo", description="Echo", strict=True)
        chat_format = _tools_to_chat_format([tool])
        responses_format = _tools_to_responses_format([tool])
        assert chat_format[0]["function"]["strict"] is True
        assert responses_format[0]["strict"] is True

    def test_message_contract_rejects_malformed_tool_calls(self) -> None:
        with pytest.raises(ValidationError):
            Message(
                role="assistant",
                tool_calls=[ToolCall.model_validate({"id": "", "function": {"name": "", "arguments": {}}})],
            )
        with pytest.raises(ValidationError, match="tool_call_id"):
            Message(role="tool", content="result")

    def test_message_tool_calls_are_normalized_to_models_with_legacy_access(self) -> None:
        message = Message(
            role="assistant",
            tool_calls=[
                ToolCall.model_validate(
                    {"id": "tc-1", "function": {"name": "echo", "arguments": {"x": 1}}}
                )
            ],
        )

        assert message.tool_calls is not None
        assert message.tool_calls[0].function.name == "echo"
        assert message.tool_calls[0]["function"]["arguments"] == '{"x": 1}'

    def test_compaction_summary_requires_canonical_marker(self) -> None:
        with pytest.raises(ValidationError, match=COMPACTION_SUMMARY_MARKER):
            CompactionSummary(marker="legacy-marker")

    def test_execution_report_rejects_invalid_timestamp_fields(self) -> None:
        with pytest.raises(ValidationError, match="ISO-8601"):
            ExecutionReport(started_at="not-a-date")

    def test_shell_command_plan_approval_status_transitions_are_enforced(self) -> None:
        plan = ShellCommandPlan(command="echo ok")
        plan.transition_to("approved").transition_to("executed")
        assert plan.approval_status == "executed"
        with pytest.raises(ValueError, match="invalid shell approval_status transition"):
            plan.transition_to("rejected")

    def test_agent_config_rejects_negative_compaction_limits(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(agent_id="agent", model="gpt-4o", auto_compact_threshold=-1)
        with pytest.raises(ValidationError):
            AgentConfig(agent_id="agent", model="gpt-4o", micro_compact_keep_recent=-1)
        with pytest.raises(ValidationError):
            AgentConfig(agent_id="agent", model="gpt-4o", max_tool_result_size=0)

    def test_agent_config_rejects_blank_model_names(self) -> None:
        with pytest.raises(ValidationError, match="model must be a non-empty string"):
            AgentConfig(agent_id="agent", model="   ")

    def test_agent_config_extra_setter_skips_full_revalidation(self) -> None:
        cfg = AgentConfig(agent_id="agent", model="gpt-4o")
        cfg.output_token_soft_limit = 10
        cfg.output_token_hard_limit = 5

        cfg.extra = {"trace": "legacy"}

        assert cfg.extra == {"trace": "legacy"}


class TestBatchCM027M039M041:
    def test_cancellation_context_wait_unblocks_across_loops(self) -> None:
        ctx = CancellationContext()
        finished = threading.Event()

        def _worker() -> None:
            async def _waiter() -> None:
                await ctx.wait()
                finished.set()

            asyncio.run(_waiter())

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        assert not finished.wait(timeout=0.1)
        ctx.cancel("stop")
        assert finished.wait(timeout=1.0)
        thread.join(timeout=1.0)

    def test_thinking_profile_defaults_follow_live_registry(self) -> None:
        ThinkingProfileRegistry.register("runtime_live_profile", {"enable_thinking": True})
        try:
            assert "runtime_live_profile" in THINKING_PROFILE_DEFAULTS
            cfg = AgentConfig(
                agent_id="agent",
                model="gpt-4o",
                thinking_profile=ThinkingProfilePreset.THINKING_PLANNER,
            )
            original = ThinkingProfileRegistry.get("thinking_planner")
            ThinkingProfileRegistry.register(
                "thinking_planner",
                {"temperature": 0.25, "enable_thinking": True},
            )
            try:
                resolved = cfg.resolved_with_selective_thinking()
                assert resolved.temperature == 0.25
                assert resolved.enable_thinking is True
            finally:
                if original is not None:
                    ThinkingProfileRegistry.register("thinking_planner", original)
        finally:
            ThinkingProfileRegistry.unregister("runtime_live_profile")

    def test_thinking_profile_registry_can_reset_to_defaults(self) -> None:
        ThinkingProfileRegistry.register("ephemeral", {"enable_thinking": True})
        ThinkingProfileRegistry.register("thinking_planner", {"temperature": 0.25})

        ThinkingProfileRegistry.reset_to_defaults()

        assert "ephemeral" not in THINKING_PROFILE_DEFAULTS
        assert ThinkingProfileRegistry.get("thinking_planner") == {
            "enable_thinking": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        }

    def test_resolve_effective_llm_config_rejects_none(self) -> None:
        with pytest.raises(ValueError, match="config is required"):
            resolve_effective_llm_config(None)
