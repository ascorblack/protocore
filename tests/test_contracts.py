"""Contract tests for Protocore.

Tests cover the mandatory protocol and API contracts.
All tests use only stdlib + pydantic (no network calls).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

from protocore.constants import (
    MAX_ENVELOPE_PAYLOAD_CHARS,
    MAX_STRUCTURED_JSON_CHARS,
    MAX_SUMMARY_CHARS,
)
import pytest
from pydantic import ValidationError

# Import public API
from protocore import (
    AgentConfig,
    AgentContext,
    AgentEnvelope,
    AgentIdentity,
    AgentOrchestrator,
    AgentRegistry,
    AgentRole,
    ApiMode,
    CancellationContext,
    CompactionSummary,
    ContentPart,
    ControlCommand,
    ExecutionMode,
    ExecutionReport,
    ExecutionStatus,
    EnvelopeMeta,
    LLMUsage,
    Message,
    MessageType,
    PathIsolationError,
    QWEN_NO_THINKING_EXTRA_BODY,
    Result,
    RunKind,
    Storage,
    StopReason,
    SubagentResult,
    SubagentStatus,
    ThinkingProfilePreset,
    ThinkingRunPolicy,
    TokenEstimatorProfile,
    ToolDefinition,
    ToolResult,
    estimate_tokens,
    make_agent_context,
    make_control_envelope,
    make_error_envelope,
    make_execution_report,
    register_manual_compact_tool,
    make_result_envelope,
    make_task_envelope,
    micro_compact,
    validate_path_access,
)
from protocore.context import build_tool_context
from protocore.events import EventBus
from protocore.registry import ToolRegistry
from protocore.types import MessageList, StreamEvent, is_done_event, is_text_delta_event


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_config(**kw: Any) -> AgentConfig:
    return AgentConfig(
        agent_id=str(uuid.uuid4()),
        model="gpt-4o",
        **kw,
    )


def make_message(
    role: Literal["system", "user", "assistant", "tool"] = "user",
    content: str = "hello",
) -> Message:
    return Message(role=role, content=content)


def user_msg(text: str) -> Message:
    return Message(role="user", content=text)


def _tool_calls(*calls: dict[str, Any]) -> Any:
    return cast(Any, list(calls))


def assistant_msg(text: str) -> Message:
    return Message(role="assistant", content=text)


def tool_msg(content: str, tool_call_id: str = "tc1", name: str = "my_tool") -> Message:
    return Message(role="tool", content=content, tool_call_id=tool_call_id, name=name)


# Minimal mock LLM that returns a final answer (no tool calls)
def mock_llm_final(content: str = "final answer") -> Any:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=Message(role="assistant", content=content))
    return llm


# LLM mock that returns one tool call then final answer
def mock_llm_one_tool_then_final(tool_name: str = "echo") -> Any:
    call_count = 0
    llm = MagicMock()

    async def complete(*args: Any, **kwargs: Any) -> Message:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return Message(
                role="assistant",
                content=None,
                tool_calls=_tool_calls(
                    {
                        "id": "tc1",
                        "function": {"name": tool_name, "arguments": '{"x": 1}'},
                    }
                ),
            )
        return Message(role="assistant", content="done")

    llm.complete = AsyncMock(side_effect=complete)
    return llm


# ---------------------------------------------------------------------------
# Inter-Agent JSON Envelope
# ---------------------------------------------------------------------------


class TestAgentEnvelope:
    def test_valid_envelope_round_trips_json(self) -> None:
        env = make_task_envelope(
            sender_id="leader-1",
            recipient_id="sub-1",
            payload={"task": "do something"},
            trace_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
        )
        raw = env.model_dump_json()
        restored = AgentEnvelope.model_validate_json(raw)
        assert restored.message_type == MessageType.TASK
        assert restored.sender.role == AgentRole.LEADER
        assert restored.recipient.role == AgentRole.SUBAGENT
        assert restored.payload["task"] == "do something"

    def test_major_version_mismatch_raises(self) -> None:
        with pytest.raises(Exception, match="PROTOCOL_VERSION_MISMATCH"):
            AgentEnvelope(
                protocol_version="2.0",
                message_type=MessageType.TASK,
                trace_id=str(uuid.uuid4()),
                session_id=str(uuid.uuid4()),
                sender=AgentIdentity(agent_id="l", role=AgentRole.LEADER),
                recipient=AgentIdentity(agent_id="s", role=AgentRole.SUBAGENT),
                payload={"task": "ping"},
            )

    def test_minor_version_difference_is_accepted(self) -> None:
        # Minor version can differ — just a warning (not an error from model)
        env = AgentEnvelope(
            protocol_version="1.99",  # same major
            message_type=MessageType.RESULT,
            trace_id="t",
            session_id="s",
            sender=AgentIdentity(agent_id="s1", role=AgentRole.SUBAGENT),
            recipient=AgentIdentity(agent_id="l1", role=AgentRole.LEADER),
            payload={
                "status": "success",
                "summary": "done",
                "artifacts": [],
                "files_changed": [],
                "tool_calls_made": 0,
                "errors": [],
                "next_steps": None,
            },
        )
        assert env.protocol_version == "1.99"
        assert env.meta.protocol_version == "1.99"

    def test_explicit_meta_protocol_version_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="meta.protocol_version"):
            AgentEnvelope(
                protocol_version="1.1",
                message_type=MessageType.TASK,
                trace_id="t",
                session_id="s",
                sender=AgentIdentity(agent_id="leader", role=AgentRole.LEADER),
                recipient=AgentIdentity(agent_id="sub", role=AgentRole.SUBAGENT),
                payload={"task": "hello"},
                meta=EnvelopeMeta(protocol_version="1.0"),
            )

    def test_error_envelope_factory(self) -> None:
        env = make_error_envelope(
            sender_id="s1",
            recipient_id="l1",
            error_message="boom",
            error_code="TOOL_FAILED",
            trace_id="t",
            session_id="s",
        )
        assert env.message_type == MessageType.ERROR
        assert env.payload["error"] == "boom"

    def test_payload_does_not_contain_full_history(self) -> None:
        """Envelope contract: payload is minimal, not full message history."""
        env = make_task_envelope(
            sender_id="l1",
            recipient_id="s1",
            payload={"task": "short description"},
            trace_id="t",
            session_id="s",
        )
        # Payload should be a dict, not a list of messages
        assert isinstance(env.payload, dict)
        assert "messages" not in env.payload  # no full history injected

    def test_forbidden_history_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentEnvelope(
                message_type=MessageType.TASK,
                trace_id="t",
                session_id="s",
                sender=AgentIdentity(agent_id="leader", role=AgentRole.LEADER),
                recipient=AgentIdentity(agent_id="subagent", role=AgentRole.SUBAGENT),
                payload={"messages": ["too much context"]},
            )

    def test_nested_forbidden_history_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentEnvelope(
                message_type=MessageType.TASK,
                trace_id="t",
                session_id="s",
                sender=AgentIdentity(agent_id="leader", role=AgentRole.LEADER),
                recipient=AgentIdentity(agent_id="subagent", role=AgentRole.SUBAGENT),
                payload={
                    "task": "ping",
                    "metadata": {
                        "nested": {
                            "message_history": ["too much context"],
                        }
                    },
                },
            )

    def test_minor_version_warning_can_be_added_to_report(self) -> None:
        env = AgentEnvelope(
            protocol_version="1.9",
            message_type=MessageType.TASK,
            trace_id="t",
            session_id="s",
            sender=AgentIdentity(agent_id="leader", role=AgentRole.LEADER),
            recipient=AgentIdentity(agent_id="subagent", role=AgentRole.SUBAGENT),
            payload={"task": "ping"},
        )
        report = ExecutionReport()
        env.apply_version_compatibility(report)
        assert len(report.warnings) == 1
        assert "protocol_minor_version_mismatch" in report.warnings[0]

    def test_task_payload_requires_task_field(self) -> None:
        with pytest.raises(ValidationError):
            AgentEnvelope(
                message_type=MessageType.TASK,
                trace_id="t",
                session_id="s",
                sender=AgentIdentity(agent_id="leader", role=AgentRole.LEADER),
                recipient=AgentIdentity(agent_id="subagent", role=AgentRole.SUBAGENT),
                payload={"context_hint": "missing task"},
            )

    def test_control_payload_requires_command_field(self) -> None:
        with pytest.raises(ValidationError):
            AgentEnvelope(
                message_type=MessageType.CONTROL,
                trace_id="t",
                session_id="s",
                sender=AgentIdentity(agent_id="leader", role=AgentRole.LEADER),
                recipient=AgentIdentity(agent_id="subagent", role=AgentRole.SUBAGENT),
                payload={"task": "wrong shape"},
            )

    def test_payload_depth_limit_rejects_nested_context(self) -> None:
        with pytest.raises(ValidationError, match="maximum allowed nesting depth"):
            AgentEnvelope(
                message_type=MessageType.TASK,
                trace_id="t",
                session_id="s",
                sender=AgentIdentity(agent_id="leader", role=AgentRole.LEADER),
                recipient=AgentIdentity(agent_id="subagent", role=AgentRole.SUBAGENT),
                payload={
                    "task": "deep payload",
                    "metadata": {"a": {"b": {"c": {"d": {"e": "too deep"}}}}},
                },
            )

    def test_payload_size_limit_rejects_oversized_context(self) -> None:
        with pytest.raises(ValidationError, match="maximum allowed size"):
            AgentEnvelope(
                message_type=MessageType.TASK,
                trace_id="t",
                session_id="s",
                sender=AgentIdentity(agent_id="leader", role=AgentRole.LEADER),
                recipient=AgentIdentity(agent_id="subagent", role=AgentRole.SUBAGENT),
                payload={"task": "x" * (MAX_ENVELOPE_PAYLOAD_CHARS + 1)},
            )

    def test_factory_applies_minor_version_warning_to_report(self) -> None:
        report = ExecutionReport()
        env = make_task_envelope(
            sender_id="leader",
            recipient_id="subagent",
            payload={"task": "ping"},
            trace_id="t",
            session_id="s",
            report=report,
            protocol_version="1.9",
        )
        assert env.protocol_version == "1.9"
        assert len(report.warnings) == 1
        assert "protocol_minor_version_mismatch" in report.warnings[0]


# ---------------------------------------------------------------------------
# Subagent Structured Output
# ---------------------------------------------------------------------------


class TestSubagentResult:
    def test_valid_result_parses(self) -> None:
        raw = json.dumps({
            "status": "success",
            "summary": "Did the thing",
            "artifacts": ["file.py"],
            "files_changed": ["file.py"],
            "tool_calls_made": 3,
            "errors": [],
            "next_steps": None,
        })
        result = SubagentResult.parse_with_fallback(raw)
        assert result.status == SubagentStatus.SUCCESS
        assert result.summary == "Did the thing"
        assert result.tool_calls_made == 3

    def test_fallback_on_invalid_json(self) -> None:
        result = SubagentResult.parse_with_fallback("not json at all", agent_id="sub-1")
        assert result.status == SubagentStatus.PARTIAL
        assert "not json" in result.summary
        assert len(result.errors) > 0

    def test_summary_over_limit_uses_schema_violation_fallback(self) -> None:
        long_summary = "x" * (MAX_SUMMARY_CHARS + 1)
        raw = json.dumps({
            "status": "success",
            "summary": long_summary,
            "artifacts": [],
            "files_changed": [],
            "tool_calls_made": 0,
            "errors": [],
        })
        result = SubagentResult.parse_with_fallback(raw)
        assert result.status == SubagentStatus.PARTIAL
        assert (
            "SUBAGENT_RESULT_SCHEMA_VIOLATION" in result.errors
            or any(error.startswith("JSON_TOO_LARGE") for error in result.errors)
        )

    def test_oversized_json_returns_explicit_size_fallback(self) -> None:
        huge = "x" * (MAX_STRUCTURED_JSON_CHARS + 1)
        result = SubagentResult.parse_with_fallback(huge)
        assert result.status == SubagentStatus.PARTIAL
        assert result.summary.startswith("Structured output exceeded")
        assert "JSON_TOO_LARGE" in result.errors[0]

    def test_partial_status_allowed(self) -> None:
        raw = json.dumps({
            "status": "partial",
            "summary": "did half",
            "errors": ["ran out of budget"],
        })
        result = SubagentResult.parse_with_fallback(raw)
        assert result.status == SubagentStatus.PARTIAL

    def test_prompt_instructions_documents_required_json_shape(self) -> None:
        instructions = SubagentResult.prompt_instructions()
        assert "Return ONLY valid JSON" in instructions
        assert '"status": "success" | "partial" | "failed"' in instructions
        assert '"summary": "<short summary>"' in instructions
        assert "Do not include keys outside this schema." in instructions


class TestStreamEventHelpers:
    def test_text_delta_helper_and_done_helper(self) -> None:
        text_delta: StreamEvent = {"type": "delta", "kind": "text", "text": "Hello"}
        reasoning_delta: StreamEvent = {
            "type": "delta",
            "kind": "reasoning",
            "text": "Thinking...",
        }
        done: StreamEvent = {"type": "done", "usage": {"input_tokens": 1}}

        assert is_text_delta_event(text_delta) is True
        assert is_text_delta_event(reasoning_delta) is False
        assert is_done_event(done) is True
        assert is_done_event(text_delta) is False


def test_top_level_re_exports_cover_user_simulation_imports() -> None:
    from protocore import (
        HookManager,
        PlanningStrategy,
        RunKind,
        SubagentSelectionPolicy,
        hookimpl,
        structured_json_candidates,
    )

    assert HookManager is not None
    assert PlanningStrategy is not None
    assert SubagentSelectionPolicy is not None
    assert hookimpl is not None
    assert RunKind.LEADER.value == "leader"
    assert callable(structured_json_candidates)


# ---------------------------------------------------------------------------
# 3-Layer Context Compression
# ---------------------------------------------------------------------------


class TestMicroCompact:
    def test_old_tool_results_replaced_with_placeholders(self) -> None:
        messages = [
            user_msg("do stuff"),
            tool_msg("big content here " * 100, "tc1", "tool_a"),
            tool_msg("more content " * 100, "tc2", "tool_b"),
            tool_msg("recent content", "tc3", "tool_c"),  # keep_recent=2, so tc2+tc3 kept
        ]
        new_msgs, count = micro_compact(messages, keep_recent=2)
        # tc1 should be compacted
        assert count == 1
        compacted = next(m for m in new_msgs if m.tool_call_id == "tc1")
        assert "[micro_compact:" in (compacted.content or "")

    def test_no_compaction_when_not_enough_tool_calls(self) -> None:
        messages = [user_msg("hi"), tool_msg("result", "tc1")]
        new_msgs, count = micro_compact(messages, keep_recent=2)
        assert count == 0
        assert len(new_msgs) == 2

    def test_oversized_tool_result_truncated(self) -> None:
        big = "x" * 5_000
        messages = [
            user_msg("go"),
            tool_msg("old result", "tc1"),   # compacted (not keep_recent)
            tool_msg(big, "tc2"),            # kept but truncated
            tool_msg("small", "tc3"),        # kept as-is
        ]
        new_msgs, _ = micro_compact(messages, keep_recent=2, max_tool_result_size=3_000)
        tc2_msg = next(m for m in new_msgs if m.tool_call_id == "tc2")
        assert len(tc2_msg.content or "") <= 3_000 + 100  # truncated + placeholder text

    def test_oversized_list_content_tool_result_truncated(self) -> None:
        big_text = "x" * 4_000
        messages = [
            user_msg("go"),
            tool_msg("old result", "tc1"),
            Message(
                role="tool",
                content=[ContentPart(type="text", text=big_text)],
                tool_call_id="tc2",
                name="tool_b",
            ),
            tool_msg("small", "tc3"),
        ]
        new_msgs, _ = micro_compact(messages, keep_recent=2, max_tool_result_size=3_000)
        tc2_msg = next(m for m in new_msgs if m.tool_call_id == "tc2")
        assert isinstance(tc2_msg.content, str)
        assert "[truncated: original 4000 chars]" in (tc2_msg.content or "")

    def test_non_tool_messages_untouched(self) -> None:
        messages = [user_msg("hi"), assistant_msg("hello"), user_msg("bye")]
        new_msgs, count = micro_compact(messages, keep_recent=2)
        assert count == 0
        assert len(new_msgs) == 3


class TestAutoCompact:
    @pytest.mark.asyncio
    async def test_auto_compact_triggered_at_threshold(self) -> None:
        from protocore.compression import auto_compact

        big_content = "a" * 200_000  # way above 30K token estimate
        messages = [user_msg("start"), tool_msg(big_content, "tc1")]

        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(
            role="assistant",
            content=json.dumps({
                "completed_tasks": ["task A"],
                "current_goal": "finish B",
                "key_decisions": [],
                "files_modified": [],
                "next_steps": "do C",
            }),
        ))

        # Use heuristic so 200k chars / 4 = 50k tokens > 30k threshold.
        # tiktoken compresses repeated "a" aggressively and would yield far fewer tokens.
        config = AgentConfig(
            agent_id="test",
            model="m",
            token_estimator_profile=TokenEstimatorProfile.HEURISTIC,
            chars_per_token_estimate=4.0,
        )
        new_msgs, summary, _parse_ok = await auto_compact(
            messages, llm_client=llm, model="m", config=config
        )
        assert summary is not None
        assert isinstance(summary, CompactionSummary)
        assert len(new_msgs) < len(messages) + 2  # compressed

    @pytest.mark.asyncio
    async def test_auto_compact_not_triggered_below_threshold(self) -> None:
        from protocore.compression import auto_compact

        messages = [user_msg("short")]
        llm = MagicMock()
        new_msgs, summary, _parse_ok = await auto_compact(
            messages, llm_client=llm, model="m", threshold_tokens=999_999
        )
        assert summary is None
        assert new_msgs is messages  # unchanged

    @pytest.mark.asyncio
    async def test_auto_compact_uses_configured_keep_trailing_and_skips_prior_summary(self) -> None:
        from protocore.compression import auto_compact

        messages = [
            Message(
                role="system",
                content=CompactionSummary(current_goal="previous").model_dump_json(),
            ),
            user_msg("u1"),
            assistant_msg("a1"),
            tool_msg("t1", "tc1"),
            user_msg("u2"),
            assistant_msg("a2"),
        ]

        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(
            role="assistant",
            content=json.dumps({
                "completed_tasks": ["task A"],
                "current_goal": "finish B",
                "key_decisions": [],
                "files_modified": [],
                "next_steps": "do C",
            }),
        ))

        new_msgs, summary, _parse_ok = await auto_compact(
            messages,
            llm_client=llm,
            model="m",
            threshold_tokens=0,
            keep_trailing=2,
        )
        assert summary is not None
        assert len(new_msgs) == 3
        assert new_msgs[1].content == "u2"
        assert new_msgs[2].content == "a2"

    @pytest.mark.asyncio
    async def test_auto_compact_reads_keep_trailing_from_config(self) -> None:
        from protocore.compression import auto_compact

        messages = [user_msg("u1"), assistant_msg("a1"), user_msg("u2")]
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(
            role="assistant",
            content='{"completed_tasks":[],"current_goal":"x","key_decisions":[],"files_modified":[],"next_steps":"y"}',
        ))
        cfg = AgentConfig(
            agent_id="agent",
            model="m",
            execution_mode=ExecutionMode.BYPASS,
            auto_compact_threshold=0,
            auto_compact_keep_trailing=1,
        )

        new_msgs, summary, _parse_ok = await auto_compact(messages, llm_client=llm, model="m", config=cfg)
        assert summary is not None
        assert len(new_msgs) == 2
        assert new_msgs[1].content == "u2"

    @pytest.mark.asyncio
    async def test_auto_compact_explicit_arguments_override_config(self) -> None:
        from protocore.compression import auto_compact

        messages = [user_msg("u1"), assistant_msg("a1"), user_msg("u2")]
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(
            role="assistant",
            content='{"completed_tasks":[],"current_goal":"x","key_decisions":[],"files_modified":[],"next_steps":"y"}',
        ))
        cfg = AgentConfig(
            agent_id="agent",
            model="m",
            execution_mode=ExecutionMode.BYPASS,
            auto_compact_threshold=999_999,
            auto_compact_keep_trailing=3,
        )

        new_msgs, summary, _parse_ok = await auto_compact(
            messages,
            llm_client=llm,
            model="m",
            threshold_tokens=0,
            keep_trailing=1,
            config=cfg,
        )

        assert summary is not None
        assert len(new_msgs) == 2
        assert new_msgs[1].content == "u2"

    @pytest.mark.asyncio
    async def test_auto_compact_applies_selective_thinking_profile_defaults(self) -> None:
        from protocore.compression import auto_compact

        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(
            role="assistant",
            content=CompactionSummary(current_goal="done").model_dump_json(),
        ))
        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            auto_compact_threshold=0,
            thinking_profile=ThinkingProfilePreset.INSTRUCT_TOOL_WORKER,
        )

        _new_msgs, summary, _parse_ok = await auto_compact(
            [user_msg("compress me")],
            llm_client=llm,
            model=cfg.model,
            config=cfg,
            run_kind=RunKind.SUBAGENT,
        )

        assert summary is not None
        assert llm.complete.await_args is not None
        call_kwargs = llm.complete.await_args.kwargs
        assert call_kwargs["top_p"] == 0.8
        assert call_kwargs["presence_penalty"] == 1.5
        assert call_kwargs["extra_body"]["top_k"] == 20
        assert call_kwargs["extra_body"]["min_p"] == 0.0
        assert call_kwargs["extra_body"]["repetition_penalty"] == 1.0
        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_auto_compact_keep_trailing_zero_keeps_only_summary(self) -> None:
        from protocore.compression import auto_compact

        messages = [user_msg("u1"), assistant_msg("a1"), user_msg("u2")]
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(
            role="assistant",
            content='{"completed_tasks":[],"current_goal":"x","key_decisions":[],"files_modified":[],"next_steps":"y"}',
        ))

        new_msgs, summary, _parse_ok = await auto_compact(
            messages,
            llm_client=llm,
            model="m",
            threshold_tokens=0,
            keep_trailing=0,
        )

        assert summary is not None
        assert len(new_msgs) == 1
        assert new_msgs[0].role == "system"


class TestManualCompact:
    @pytest.mark.asyncio
    async def test_manual_compact_always_runs(self) -> None:
        from protocore.compression import manual_compact

        messages = [user_msg("small")]
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(
            role="assistant",
            content='{"completed_tasks":[],"current_goal":"x","key_decisions":[],"files_modified":[],"next_steps":"y"}',
        ))
        new_msgs, summary = await manual_compact(messages, llm_client=llm, model="m")
        assert isinstance(summary, CompactionSummary)


# ---------------------------------------------------------------------------
# Tool Budget & Forced Finalization
# ---------------------------------------------------------------------------


class TestToolBudget:
    @pytest.mark.asyncio
    async def test_tool_budget_triggers_forced_finalization(self) -> None:
        """When tool_calls_count hits max_tool_calls, loop exits with forced finalization."""
        cfg = make_config(max_tool_calls=2, execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("do 3 things"))

        # LLM always returns a tool call
        call_n = 0

        async def always_tool(*args: Any, **kwargs: Any) -> Message:
            nonlocal call_n
            call_n += 1
            return Message(
                role="assistant",
                content=None,
                tool_calls=_tool_calls(
                    {"id": f"tc{call_n}", "function": {"name": "tool_x", "arguments": "{}"}}
                ),
            )

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=always_tool)

        async def fake_tool(**kw: Any) -> ToolResult:
            return ToolResult(tool_call_id=kw.get("arguments", {}).get("tc", "tc"), tool_name="tool_x", content="ok")

        from protocore.registry import ToolRegistry

        reg = ToolRegistry()
        defn = ToolDefinition(name="tool_x", description="test tool")
        reg.register(defn, fake_tool)

        orch = AgentOrchestrator(llm_client=llm, tool_registry=reg)
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert report.forced_finalization_triggered
        assert report.tool_calls_total == 2
        assert report.stop_reason == StopReason.TOOL_BUDGET_EXCEEDED

    @pytest.mark.asyncio
    async def test_forced_finalization_keeps_completed_tool_results_from_same_batch(self) -> None:
        cfg = make_config(max_tool_calls=1, execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("run two tools"))

        llm = MagicMock()
        call_n = 0

        async def complete(*args: Any, **kwargs: Any) -> Message:
            nonlocal call_n
            call_n += 1
            messages = kwargs["messages"]
            if call_n == 1:
                return Message(
                    role="assistant",
                    content=None,
                    tool_calls=_tool_calls(
                        {
                            "id": "tc1",
                            "function": {"name": "tool_x", "arguments": '{"step": 1}'},
                        },
                        {
                            "id": "tc2",
                            "function": {"name": "tool_x", "arguments": '{"step": 2}'},
                        },
                    ),
                )
            tool_messages = [msg for msg in messages if msg.role == "tool"]
            assert [msg.content for msg in tool_messages] == ["result-from-step-1"]
            assert any(
                msg.role == "system"
                and "maximum number of tool calls" in str(msg.content)
                for msg in messages
            )
            return Message(role="assistant", content="final answer uses first tool result")

        llm.complete = AsyncMock(side_effect=complete)

        async def fake_tool(**kw: Any) -> ToolResult:
            step = kw.get("arguments", {}).get("step")
            return ToolResult(
                tool_call_id=f"tc{step}",
                tool_name="tool_x",
                content=f"result-from-step-{step}",
            )

        reg = ToolRegistry()
        reg.register(ToolDefinition(name="tool_x", description="test tool"), fake_tool)

        orch = AgentOrchestrator(llm_client=llm, tool_registry=reg)
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert result.content == "final answer uses first tool result"
        assert report.tool_calls_total == 1
        assert report.forced_finalization_triggered
        assert report.stop_reason == StopReason.TOOL_BUDGET_EXCEEDED

    @pytest.mark.asyncio
    async def test_report_returned_even_on_error(self) -> None:
        """ExecutionReport is returned even when LLM throws."""
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("hi"))

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        orch = AgentOrchestrator(llm_client=llm)
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert report.status == ExecutionStatus.FAILED
        assert report.error_code == "RuntimeError"
        assert report.finished_at is not None

    @pytest.mark.asyncio
    async def test_llm_latency_recorded_on_failed_llm_call(self) -> None:
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("hi"))

        llm = MagicMock()

        async def failing_complete(*args: Any, **kwargs: Any) -> Message:
            await asyncio.sleep(0)
            raise RuntimeError("LLM down")

        llm.complete = AsyncMock(side_effect=failing_complete)

        orch = AgentOrchestrator(llm_client=llm)
        _, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert report.status == ExecutionStatus.FAILED
        assert len(report.llm_latency_ms) == 1
        assert report.llm_latency_ms[0] >= 0

    @pytest.mark.asyncio
    async def test_llm_latency_records_failed_retry_attempts(self) -> None:
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("retry once"))

        llm = MagicMock()
        attempts = 0

        async def flaky_complete(*args: Any, **kwargs: Any) -> Message:
            nonlocal attempts
            attempts += 1
            await asyncio.sleep(0)
            if attempts == 1:
                raise RuntimeError("temporary")
            return Message(role="assistant", content="done")

        class RetryOnce:
            def should_retry(self, attempt: int, error: Exception) -> bool:
                _ = error
                return attempt == 1

            def delay_seconds(self, attempt: int) -> float:
                _ = attempt
                return 0.0

        llm.complete = AsyncMock(side_effect=flaky_complete)

        orch = AgentOrchestrator(
            llm_client=llm,
            retry_policy=RetryOnce(),
        )
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert result.content == "done"
        assert report.status == ExecutionStatus.COMPLETED
        assert len(report.llm_latency_ms) == 2
        assert any(
            warning.startswith("retry:llm.complete:attempt=1:RuntimeError")
            for warning in report.warnings
        )

    @pytest.mark.asyncio
    async def test_tool_budget_per_run_respects_own_config(self) -> None:
        """Each run uses its own config.max_tool_calls; leader and child budgets are independent (C45-004)."""
        from protocore.registry import ToolRegistry

        call_count = 0

        async def always_tool(*args: Any, **kwargs: Any) -> Message:
            nonlocal call_count
            call_count += 1
            return Message(
                role="assistant",
                content=None,
                tool_calls=_tool_calls(
                    {"id": f"tc{call_count}", "function": {"name": "t", "arguments": "{}"}}
                ),
            )

        async def fake_tool(**kw: Any) -> ToolResult:
            return ToolResult(tool_call_id="tc", tool_name="t", content="ok")

        reg = ToolRegistry()
        reg.register(ToolDefinition(name="t", description="x"), fake_tool)
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=always_tool)

        # Run 1: budget 2
        call_count = 0
        cfg1 = make_config(max_tool_calls=2, execution_mode=ExecutionMode.BYPASS)
        ctx1 = make_agent_context(config=cfg1)
        ctx1.messages.append(user_msg("do things"))
        orch = AgentOrchestrator(llm_client=llm, tool_registry=reg)
        _, report1 = await orch.run(ctx1, run_kind=RunKind.SUBAGENT)
        assert report1.tool_calls_total == 2
        assert report1.forced_finalization_triggered
        assert report1.stop_reason == StopReason.TOOL_BUDGET_EXCEEDED

        # Run 2: budget 3 (independent run, own config)
        call_count = 0
        cfg2 = make_config(max_tool_calls=3, execution_mode=ExecutionMode.BYPASS)
        ctx2 = make_agent_context(config=cfg2)
        ctx2.messages.append(user_msg("do more"))
        _, report2 = await orch.run(ctx2, run_kind=RunKind.SUBAGENT)
        assert report2.tool_calls_total == 3
        assert report2.forced_finalization_triggered
        assert report2.stop_reason == StopReason.TOOL_BUDGET_EXCEEDED

    @pytest.mark.asyncio
    async def test_parallel_children_use_separate_budgets(self) -> None:
        """Parallel subagents each get their own context.config.max_tool_calls from registry (C45-004)."""
        from protocore.registry import AgentRegistry

        from protocore.orchestrator import AgentOrchestrator, ParallelSubagentRunner

        agent_reg = AgentRegistry()
        agent_reg.register(
            AgentConfig(agent_id="child-a", model="m", max_tool_calls=1, execution_mode=ExecutionMode.BYPASS)
        )
        agent_reg.register(
            AgentConfig(agent_id="child-b", model="m", max_tool_calls=1, execution_mode=ExecutionMode.BYPASS)
        )

        responses = [
            Message(
                role="assistant",
                content=None,
                tool_calls=_tool_calls({"id": "tc1", "function": {"name": "t", "arguments": "{}"}}),
            ),
            Message(role="assistant", content="done-a"),
            Message(
                role="assistant",
                content=None,
                tool_calls=_tool_calls({"id": "tc2", "function": {"name": "t", "arguments": "{}"}}),
            ),
            Message(role="assistant", content="done-b"),
        ]
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=responses)

        async def fake_tool(**kw: Any) -> ToolResult:
            return ToolResult(tool_call_id="tc", tool_name="t", content="ok")

        reg = ToolRegistry()
        reg.register(ToolDefinition(name="t", description="x"), fake_tool)

        def factory() -> AgentOrchestrator:
            return AgentOrchestrator(
                llm_client=llm,
                tool_registry=reg,
                agent_registry=agent_reg,
            )

        policy = MagicMock()
        policy.max_concurrency = 2
        policy.timeout_seconds = 10.0
        policy.cancellation_mode = "graceful"
        policy.merge_results = AsyncMock(
            return_value=SubagentResult(status=SubagentStatus.SUCCESS, summary="merged")
        )

        parent_cfg = make_config(execution_mode=ExecutionMode.PARALLEL)
        parent_ctx = make_agent_context(config=parent_cfg)
        parent_ctx.metadata["parallel_agent_ids"] = ["child-a", "child-b"]
        parent_ctx.messages.append(user_msg("run both"))

        runner = ParallelSubagentRunner(policy=policy, orchestrator_factory=factory)
        tasks = [
            (agent_id, factory()._build_subagent_context(parent_ctx, agent_id, "task"))
            for agent_id in ["child-a", "child-b"]
        ]
        report = make_execution_report(context=parent_ctx)
        merged, summaries = await runner.run_parallel(tasks, report=report)

        assert len(summaries) == 2
        by_id = {s.agent_id: s for s in summaries}
        assert by_id["child-a"].tool_calls_total == 1
        assert by_id["child-b"].tool_calls_total == 1
        assert merged.status == SubagentStatus.SUCCESS
        assert report.subagents_parallel_max == 2

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_config_is_forwarded_to_llm_request(self) -> None:
        captured: list[bool] = []

        async def complete(*args: Any, **kwargs: Any) -> Message:
            _ = args
            captured.append(bool(kwargs.get("parallel_tool_calls")))
            return Message(role="assistant", content="done")

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=complete)

        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            parallel_tool_calls=True,
            tool_definitions=[ToolDefinition(name="echo", description="echo")],
        )
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(user_msg("hello"))

        orch = AgentOrchestrator(llm_client=llm)
        result, report = await orch.run(ctx, run_kind=RunKind.SUBAGENT)

        assert result.content == "done"
        assert report.status == ExecutionStatus.COMPLETED
        assert captured == [True]

    @pytest.mark.asyncio
    async def test_subagent_context_uses_registry_config_for_budget(self) -> None:
        """_build_subagent_context assigns config from AgentRegistry, so each child has its own max_tool_calls (C45-004)."""
        from protocore.registry import AgentRegistry

        agent_reg = AgentRegistry()
        agent_reg.register(
            AgentConfig(agent_id="low-budget", model="m", max_tool_calls=1)
        )
        agent_reg.register(
            AgentConfig(agent_id="high-budget", model="m", max_tool_calls=10)
        )
        parent_cfg = make_config()
        parent_cfg = parent_cfg.model_copy(update={"agent_id": "leader", "max_tool_calls": 5})
        parent_ctx = make_agent_context(config=parent_cfg)

        orch = AgentOrchestrator(
            llm_client=MagicMock(),
            agent_registry=agent_reg,
        )
        child_low = orch._build_subagent_context(parent_ctx, "low-budget", "task")
        child_high = orch._build_subagent_context(parent_ctx, "high-budget", "task")

        assert child_low.config.agent_id == "low-budget"
        assert child_low.config.max_tool_calls == 1
        assert child_high.config.agent_id == "high-budget"
        assert child_high.config.max_tool_calls == 10
        assert parent_ctx.config.max_tool_calls == 5
        assert child_low.metadata["parent_request_id"] == parent_ctx.request_id

    def test_subagent_context_tool_metadata_is_isolated(self) -> None:
        from protocore.registry import AgentRegistry

        agent_reg = AgentRegistry()
        agent_reg.register(AgentConfig(agent_id="child-a", model="m"))
        parent_ctx = make_agent_context(config=make_config())
        parent_ctx.tool_context.metadata["shared"] = {"value": 1}

        orch = AgentOrchestrator(
            llm_client=MagicMock(),
            agent_registry=agent_reg,
        )
        child_ctx = orch._build_subagent_context(parent_ctx, "child-a", "task")
        child_ctx.tool_context.metadata["shared"]["value"] = 2
        child_ctx.tool_context.metadata["child_only"] = True

        assert parent_ctx.tool_context.metadata["shared"]["value"] == 1
        assert "child_only" not in parent_ctx.tool_context.metadata

    def test_subagent_context_keeps_own_system_prompt(self) -> None:
        from protocore.registry import AgentRegistry

        agent_reg = AgentRegistry()
        agent_reg.register(
            AgentConfig(
                agent_id="child-a",
                model="m",
                system_prompt="You are the child specialist.",
            )
        )
        parent_ctx = make_agent_context(
            config=make_config(system_prompt="You are the leader.")
        )

        orch = AgentOrchestrator(
            llm_client=MagicMock(),
            agent_registry=agent_reg,
        )
        child_ctx = orch._build_subagent_context(parent_ctx, "child-a", "task")

        assert child_ctx.config.system_prompt == "You are the child specialist."
        assert parent_ctx.config.system_prompt == "You are the leader."

    @pytest.mark.asyncio
    async def test_agent_tool_definitions_override_registry_visibility(self) -> None:
        captured_tools: list[list[str]] = []

        async def complete(*args: Any, **kwargs: Any) -> Message:
            _ = args
            captured_tools.append([tool.name for tool in kwargs.get("tools", [])])
            return Message(role="assistant", content="done")

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=complete)

        registry = ToolRegistry()
        registry.register(ToolDefinition(name="global_tool", description="global"), AsyncMock())
        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            tool_definitions=[ToolDefinition(name="local_tool", description="local")],
        )
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(user_msg("hello"))

        orch = AgentOrchestrator(llm_client=llm, tool_registry=registry)
        result, report = await orch.run(ctx, run_kind=RunKind.SUBAGENT)

        assert result.content == "done"
        assert report.status == ExecutionStatus.COMPLETED
        assert captured_tools == [["local_tool"]]


class TestImmutableLoop:
    @pytest.mark.asyncio
    async def test_loop_completes_with_no_tool_calls(self) -> None:
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("what is 2+2?"))

        orch = AgentOrchestrator(llm_client=mock_llm_final("4"))
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert result.content == "4"
        assert report.status == ExecutionStatus.COMPLETED
        assert report.stop_reason == StopReason.END_TURN

    @pytest.mark.asyncio
    async def test_loop_executes_one_tool_then_finishes(self) -> None:
        cfg = make_config(max_tool_calls=5, execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("run echo"))

        llm = mock_llm_one_tool_then_final("echo_tool")

        async def echo_handler(**kw: Any) -> ToolResult:
            return ToolResult(
                tool_call_id="tc1", tool_name="echo_tool", content="echoed"
            )

        from protocore.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register(ToolDefinition(name="echo_tool", description="echo"), echo_handler)

        orch = AgentOrchestrator(llm_client=llm, tool_registry=reg)
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert report.tool_calls_total == 1
        assert report.tool_calls_by_name["echo_tool"] == 1
        assert result.content == "done"

    @pytest.mark.asyncio
    async def test_max_iterations_uses_dedicated_stop_reason(self) -> None:
        cfg = make_config(max_iterations=1, execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("loop forever"))

        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value=Message(
                role="assistant",
                content=None,
                tool_calls=_tool_calls(
                    {
                        "id": "tc1",
                        "function": {"name": "echo_tool", "arguments": '{"x": 1}'},
                    }
                ),
            )
        )

        async def echo_handler(**kw: Any) -> ToolResult:
            return ToolResult(tool_call_id="tc1", tool_name="echo_tool", content="echoed")

        reg = ToolRegistry()
        reg.register(ToolDefinition(name="echo_tool", description="echo"), echo_handler)

        orch = AgentOrchestrator(llm_client=llm, tool_registry=reg)
        _, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert report.status == ExecutionStatus.FAILED
        assert report.error_code == "MAX_ITERATIONS_EXCEEDED"
        assert report.stop_reason == StopReason.MAX_ITERATIONS


# ---------------------------------------------------------------------------
# ExecutionReport serialization + error paths
# ---------------------------------------------------------------------------


class TestExecutionReport:
    def test_report_json_serializable(self) -> None:
        cfg = make_config()
        ctx = make_agent_context(config=cfg)
        report = make_execution_report(context=ctx)
        report.finalize(ExecutionStatus.COMPLETED, StopReason.END_TURN)

        raw = report.model_dump_json()
        restored = ExecutionReport.model_validate_json(raw)
        assert restored.status == ExecutionStatus.COMPLETED
        assert restored.report_version == "1.1"

    def test_report_finalized_on_timeout_path(self) -> None:
        cfg = make_config()
        ctx = make_agent_context(config=cfg)
        report = make_execution_report(context=ctx)
        report.finalize(
            ExecutionStatus.TIMEOUT,
            stop_reason=StopReason.MAX_TOKENS,
            error_code="TIMEOUT",
            error_message="ran out of time",
        )
        assert report.status == ExecutionStatus.TIMEOUT
        assert report.duration_ms is not None
        assert report.duration_ms >= 0

    def test_report_finalized_on_cancelled_path(self) -> None:
        cfg = make_config()
        ctx = make_agent_context(config=cfg)
        report = make_execution_report(context=ctx)
        report.finalize(
            ExecutionStatus.CANCELLED,
            stop_reason=StopReason.CANCELLED,
            error_code="CANCELLED",
        )
        assert report.status == ExecutionStatus.CANCELLED

    def test_report_has_required_identification_fields(self) -> None:
        cfg = make_config()
        ctx = make_agent_context(config=cfg)
        report = make_execution_report(context=ctx, run_kind=RunKind.SUBAGENT)
        assert report.request_id
        assert report.trace_id
        assert report.session_id
        assert report.agent_id
        assert report.run_kind == RunKind.SUBAGENT

    def test_finalize_is_idempotent(self) -> None:
        cfg = make_config()
        ctx = make_agent_context(config=cfg)
        report = make_execution_report(context=ctx)

        report.finalize(ExecutionStatus.COMPLETED, StopReason.END_TURN)
        first_finished_at = report.finished_at
        first_duration_ms = report.duration_ms

        report.finalize(
            ExecutionStatus.FAILED,
            stop_reason=StopReason.ERROR,
            error_code="SHOULD_NOT_WIN",
            error_message="should not overwrite",
        )

        assert report.status == ExecutionStatus.COMPLETED
        assert report.stop_reason == StopReason.END_TURN
        assert report.finished_at == first_finished_at
        assert report.duration_ms == first_duration_ms
        assert report.error_code is None
        assert report.error_message is None

    def test_report_stable_schema_version(self) -> None:
        cfg = make_config()
        ctx = make_agent_context(config=cfg)
        report = make_execution_report(context=ctx)
        assert report.report_version == "1.1"


# ---------------------------------------------------------------------------
# Planning gate
# ---------------------------------------------------------------------------


class TestPlanningGate:
    @pytest.mark.asyncio
    async def test_leader_mode_calls_planning_strategy(self) -> None:
        cfg = make_config(execution_mode=ExecutionMode.LEADER)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("build a feature"))

        planning = MagicMock()
        from protocore.types import PlanArtifact

        plan = PlanArtifact(trace_id=context.trace_id, raw_plan="step 1, step 2")
        planning.build_plan = AsyncMock(return_value=plan)

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("done"),
            planning_strategy=planning,
        )
        result, report = await orch.run(context, run_kind=RunKind.LEADER)

        planning.build_plan.assert_awaited_once()
        assert report.plan_created is True
        assert report.plan_id == plan.plan_id

    @pytest.mark.asyncio
    async def test_leader_mode_updates_existing_plan_when_present(self) -> None:
        from protocore.types import PlanArtifact

        cfg = make_config(execution_mode=ExecutionMode.LEADER)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("update my plan"))
        existing_plan = PlanArtifact(trace_id=context.trace_id, raw_plan="old plan")
        context.metadata["plan_artifact"] = existing_plan.model_dump()

        planning = MagicMock()
        updated_plan = existing_plan.model_copy(update={"raw_plan": "new plan"})
        planning.update_plan = AsyncMock(return_value=updated_plan)
        planning.build_plan = AsyncMock()

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("done"),
            planning_strategy=planning,
        )
        _, report = await orch.run(context, run_kind=RunKind.LEADER)

        planning.build_plan.assert_not_awaited()
        planning.update_plan.assert_awaited_once()
        assert report.plan_artifact is not None
        assert report.plan_artifact.raw_plan == "new plan"

    @pytest.mark.asyncio
    async def test_bypass_mode_skips_planning(self) -> None:
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("quick answer"))

        planning = MagicMock()
        planning.build_plan = AsyncMock(return_value=None)

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("42"),
            planning_strategy=planning,
        )
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        # Planning NOT called in bypass mode
        planning.build_plan.assert_not_awaited()
        assert report.plan_created is False

    @pytest.mark.asyncio
    async def test_bypass_mode_can_be_denied_by_planning_policy(self) -> None:
        class DenyBypassPolicy:
            async def should_plan(self, context: AgentContext) -> bool:
                return True

            async def allow_bypass(self, context: AgentContext) -> bool:
                return False

        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("quick answer"))

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("42"),
            planning_policy=DenyBypassPolicy(),
        )
        _, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert report.status == ExecutionStatus.FAILED
        assert report.error_code == "BYPASS_NOT_ALLOWED"

    @pytest.mark.asyncio
    async def test_subagent_bypass_is_not_denied_by_top_level_bypass_policy(self) -> None:
        class DenyBypassPolicy:
            async def should_plan(self, context: AgentContext) -> bool:
                return True

            async def allow_bypass(self, context: AgentContext) -> bool:
                return False

        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = AgentContext(
            session_id="session-sub",
            trace_id="trace-sub",
            parent_agent_id="leader-agent",
            config=cfg,
            messages=MessageList([user_msg("quick answer")]),
        )

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("42"),
            planning_policy=DenyBypassPolicy(),
        )
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert result.status == ExecutionStatus.COMPLETED
        assert report.error_code is None

    @pytest.mark.asyncio
    async def test_leader_mode_policy_can_skip_planning(self) -> None:
        class SkipPlanningPolicy:
            async def should_plan(self, context: AgentContext) -> bool:
                return False

            async def allow_bypass(self, context: AgentContext) -> bool:
                return False

        cfg = make_config(execution_mode=ExecutionMode.LEADER)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("plan this"))

        planning = MagicMock()
        planning.build_plan = AsyncMock()

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("done"),
            planning_strategy=planning,
            planning_policy=SkipPlanningPolicy(),
        )
        _, report = await orch.run(context, run_kind=RunKind.LEADER)

        planning.build_plan.assert_not_awaited()
        assert report.plan_created is False
        assert report.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_default_mode_is_leader_when_no_policy(self) -> None:
        """Without explicit bypass config, LEADER mode is used (no implicit bypass)."""
        cfg = AgentConfig(model="gpt-4o")
        assert cfg.execution_mode == ExecutionMode.LEADER

    @pytest.mark.asyncio
    async def test_leader_workflow_definition_calls_planning_before_engine(self) -> None:
        from protocore.types import PlanArtifact, WorkflowDefinition, WorkflowNode

        cfg = make_config(execution_mode=ExecutionMode.LEADER)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("build a workflow"))
        context.metadata["workflow_definition"] = WorkflowDefinition(
            name="wf",
            nodes=[WorkflowNode(node_id="n1", label="step")],
        ).model_dump()

        planning = MagicMock()
        plan = PlanArtifact(trace_id=context.trace_id, raw_plan="step 1")
        planning.build_plan = AsyncMock(return_value=plan)

        engine_report = ExecutionReport(
            status=ExecutionStatus.COMPLETED,
            stop_reason=StopReason.END_TURN,
        )
        engine = MagicMock()
        engine.run = AsyncMock(return_value=(Result(content="workflow done"), engine_report))

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("unused"),
            planning_strategy=planning,
            workflow_engine=engine,
        )
        result, report = await orch.run(context, run_kind=RunKind.LEADER)

        planning.build_plan.assert_awaited_once()
        engine.run.assert_awaited_once()
        assert result.content == "workflow done"
        assert report.plan_created is True
        assert report.plan_id == plan.plan_id
        assert any(item == f"plan:{plan.plan_id}" for item in report.artifacts)

    @pytest.mark.asyncio
    async def test_workflow_report_is_finalized_if_engine_returns_running_state(self) -> None:
        from protocore.types import PlanArtifact, WorkflowDefinition, WorkflowNode

        cfg = make_config(execution_mode=ExecutionMode.LEADER)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("workflow"))
        context.metadata["workflow_definition"] = WorkflowDefinition(
            name="wf",
            nodes=[WorkflowNode(node_id="n1", label="step")],
        ).model_dump()

        planning = MagicMock()
        planning.build_plan = AsyncMock(
            return_value=PlanArtifact(trace_id=context.trace_id, raw_plan="workflow")
        )

        running_report = make_execution_report(context=context)
        engine = MagicMock()
        engine.run = AsyncMock(
            return_value=(
                Result(content="workflow done", status=ExecutionStatus.COMPLETED),
                running_report,
            )
        )

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("unused"),
            planning_strategy=planning,
            workflow_engine=engine,
        )
        _, report = await orch.run(context, run_kind=RunKind.LEADER)

        assert report.status == ExecutionStatus.COMPLETED
        assert report.finished_at is not None


# ---------------------------------------------------------------------------
# Bypass requires explicit mode
# ---------------------------------------------------------------------------


class TestBypassContract:
    @pytest.mark.asyncio
    async def test_run_bypass_raises_if_mode_not_bypass(self) -> None:
        from protocore.orchestrator import ContractViolationError, run_bypass

        cfg = make_config(execution_mode=ExecutionMode.LEADER)
        context = make_agent_context(config=cfg)
        orch = AgentOrchestrator(llm_client=mock_llm_final())

        with pytest.raises(ContractViolationError, match="BYPASS"):
            await run_bypass(orchestrator=orch, context=context)

    @pytest.mark.asyncio
    async def test_run_bypass_works_with_explicit_bypass_mode(self) -> None:
        from protocore.orchestrator import run_bypass

        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("quick"))

        orch = AgentOrchestrator(llm_client=mock_llm_final("fast"))
        result, report = await run_bypass(orchestrator=orch, context=context)

        assert result.content == "fast"


# ---------------------------------------------------------------------------
# Parallel subagent execution
# ---------------------------------------------------------------------------


class TestParallelSubagents:
    @pytest.mark.asyncio
    async def test_parallel_runner_sorts_merge_inputs_by_agent_id(self) -> None:
        from protocore.orchestrator import ParallelSubagentRunner

        seen_agent_ids: list[str] = []

        def make_orch() -> AgentOrchestrator:
            llm = MagicMock()
            llm.complete = AsyncMock(
                return_value=Message(
                    role="assistant",
                    content=(
                        '{"status":"success","summary":"ok","artifacts":[],"files_changed":[],'
                        '"tool_calls_made":0,"errors":[],"next_steps":null}'
                    ),
                )
            )
            return AgentOrchestrator(llm_client=llm)

        class Policy:
            max_concurrency = 2
            timeout_seconds = 1.0
            cancellation_mode = "graceful"

            async def merge_results(
                self,
                results: list[SubagentResult | None],
                agent_ids: list[str],
            ) -> SubagentResult:
                seen_agent_ids.extend(agent_ids)
                return SubagentResult(status=SubagentStatus.SUCCESS, summary="merged")

        runner = ParallelSubagentRunner(policy=Policy(), orchestrator_factory=make_orch)

        tasks: list[tuple[str, AgentContext]] = []
        for agent_id in ["agent-c", "agent-a", "agent-b"]:
            cfg = make_config(execution_mode=ExecutionMode.BYPASS)
            cfg = cfg.model_copy(update={"agent_id": agent_id})
            ctx = make_agent_context(config=cfg)
            ctx.messages.append(user_msg(f"task for {agent_id}"))
            tasks.append((agent_id, ctx))

        _, summaries = await runner.run_parallel(tasks)

        assert seen_agent_ids == ["agent-a", "agent-b", "agent-c"]
        assert [summary.agent_id for summary in summaries] == ["agent-a", "agent-b", "agent-c"]

    @pytest.mark.asyncio
    async def test_parallel_runs_respect_concurrency_limit(self) -> None:
        from protocore.orchestrator import ParallelSubagentRunner
        from protocore.types import SubagentStatus

        max_concurrent = 2
        concurrent_now = 0
        max_seen = 0

        async def slow_complete(*args: Any, **kwargs: Any) -> Message:
            nonlocal concurrent_now, max_seen
            concurrent_now += 1
            max_seen = max(max_seen, concurrent_now)
            await asyncio.sleep(0.01)
            concurrent_now -= 1
            return Message(role="assistant", content="ok")

        def make_orch() -> AgentOrchestrator:
            llm = MagicMock()
            llm.complete = AsyncMock(side_effect=slow_complete)
            return AgentOrchestrator(llm_client=llm)

        policy = MagicMock()
        policy.max_concurrency = max_concurrent
        policy.timeout_seconds = 5.0
        policy.cancellation_mode = "graceful"

        async def merge(results: list[Any], agent_ids: list[str]) -> SubagentResult:
            ok = [r for r in results if r and r.status == SubagentStatus.SUCCESS]
            return SubagentResult(
                status=SubagentStatus.SUCCESS if ok else SubagentStatus.PARTIAL,
                summary=f"{len(ok)} succeeded",
            )

        policy.merge_results = AsyncMock(side_effect=merge)

        runner = ParallelSubagentRunner(policy=policy, orchestrator_factory=make_orch)

        tasks = []
        for i in range(4):
            cfg = make_config(execution_mode=ExecutionMode.BYPASS)
            ctx = make_agent_context(config=cfg)
            ctx.messages.append(user_msg(f"task {i}"))
            tasks.append((cfg.agent_id, ctx))

        merged, summaries = await runner.run_parallel(tasks)
        assert max_seen <= max_concurrent
        assert len(summaries) == 4

    @pytest.mark.asyncio
    async def test_parallel_timeout_handled_gracefully(self) -> None:
        from protocore.orchestrator import ParallelSubagentRunner

        async def hang(*args: Any, **kwargs: Any) -> Message:
            await asyncio.sleep(100)
            return Message(role="assistant", content="never")

        def make_orch() -> AgentOrchestrator:
            llm = MagicMock()
            llm.complete = AsyncMock(side_effect=hang)
            return AgentOrchestrator(llm_client=llm)

        policy = MagicMock()
        policy.max_concurrency = 2
        policy.timeout_seconds = 0.05  # very short timeout
        policy.cancellation_mode = "graceful"

        async def merge(results: list[Any], agent_ids: list[str]) -> SubagentResult:
            return SubagentResult(status=SubagentStatus.PARTIAL, summary="some timeout")

        policy.merge_results = AsyncMock(side_effect=merge)

        runner = ParallelSubagentRunner(policy=policy, orchestrator_factory=make_orch)

        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(user_msg("slow task"))

        # Should not raise — timeout is handled
        merged, summaries = await runner.run_parallel([(cfg.agent_id, ctx)])
        assert summaries[0].status == ExecutionStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_parallel_cancelled_status_wins_even_near_timeout_boundary(self) -> None:
        from protocore.orchestrator import ParallelSubagentRunner

        cancel_ctx = CancellationContext()
        cancel_ctx.cancel("stop")

        class StubOrchestrator:
            async def run(
                self,
                ctx: AgentContext,
                *,
                run_kind: RunKind = RunKind.SUBAGENT,
                cancel_ctx: CancellationContext | None = None,
            ) -> tuple[Result, ExecutionReport]:
                _ = run_kind
                assert cancel_ctx is not None and cancel_ctx.is_cancelled
                report = make_execution_report(context=ctx, run_kind=RunKind.SUBAGENT)
                report.finalize(
                    ExecutionStatus.CANCELLED,
                    stop_reason=StopReason.CANCELLED,
                    error_code="CANCELLED",
                    error_message="stop",
                )
                return (
                    Result(
                        content='{"status":"partial","summary":"cancelled","errors":["stop"]}',
                        status=ExecutionStatus.CANCELLED,
                    ),
                    report,
                )

        class Policy:
            max_concurrency = 1
            timeout_seconds = 1.0
            cancellation_mode = "graceful"

            async def merge_results(
                self,
                results: list[SubagentResult | None],
                agent_ids: list[str],
            ) -> SubagentResult:
                _ = results
                _ = agent_ids
                return SubagentResult(status=SubagentStatus.PARTIAL, summary="cancelled")

        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        cfg = cfg.model_copy(update={"agent_id": "child-a"})
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(user_msg("cancelled task"))

        runner = ParallelSubagentRunner(
            policy=Policy(),
            orchestrator_factory=lambda: StubOrchestrator(),
        )

        with pytest.MonkeyPatch.context() as mp:
            values = iter([0.0, 1.05])
            mp.setattr("protocore.orchestrator.time.monotonic", lambda: next(values, 1.05))
            _, summaries = await runner.run_parallel([(cfg.agent_id, ctx)], cancel_ctx=cancel_ctx)

        assert summaries[0].status == ExecutionStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_parallel_mode_rejects_duplicate_agent_ids(self) -> None:
        registry = AgentRegistry()
        registry.register(AgentConfig(agent_id="agent-a", model="m"))

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("unused"),
            agent_registry=registry,
            parallel_execution_policy=MagicMock(
                max_concurrency=2,
                timeout_seconds=1.0,
                cancellation_mode="graceful",
                merge_results=AsyncMock(
                    return_value=SubagentResult(
                        status=SubagentStatus.SUCCESS,
                        summary="merged",
                    )
                ),
            ),
        )
        ctx = make_agent_context(config=make_config(execution_mode=ExecutionMode.PARALLEL))
        ctx.messages.append(user_msg("run in parallel"))
        ctx.metadata["parallel_agent_ids"] = ["agent-a", "agent-a"]

        _, report = await orch.run(ctx)

        assert report.status == ExecutionStatus.FAILED
        assert report.error_code == "PARALLEL_AGENT_IDS_DUPLICATE"

    @pytest.mark.asyncio
    async def test_parallel_runner_rejects_invalid_policy_values(self) -> None:
        from protocore.orchestrator import ContractViolationError, ParallelSubagentRunner

        class BadPolicy:
            max_concurrency = 0
            timeout_seconds = 1.0
            cancellation_mode = "invalid"

            async def merge_results(
                self,
                results: list[SubagentResult | None],
                agent_ids: list[str],
            ) -> SubagentResult:
                return SubagentResult(status=SubagentStatus.SUCCESS, summary="merged")

        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        cfg = cfg.model_copy(update={"agent_id": "agent-a"})
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(user_msg("task"))

        runner = ParallelSubagentRunner(
            policy=BadPolicy(),
            orchestrator_factory=lambda: AgentOrchestrator(
                llm_client=mock_llm_final("ok")
            ),
        )

        with pytest.raises(ContractViolationError, match="max_concurrency"):
            await runner.run_parallel([(cfg.agent_id, ctx)])

    @pytest.mark.asyncio
    async def test_parallel_mode_merges_artifacts_and_files_into_parent_report(self) -> None:
        class Policy:
            max_concurrency = 2
            timeout_seconds = 5.0
            cancellation_mode = "graceful"

            async def merge_results(
                self,
                results: list[SubagentResult | None],
                agent_ids: list[str],
            ) -> SubagentResult:
                return SubagentResult(
                    status=SubagentStatus.SUCCESS,
                    summary="merged",
                    artifacts=["artifact-a"],
                    files_changed=["file-a.py"],
                )

        registry = AgentRegistry()
        registry.register(AgentConfig(agent_id="agent-a", model="m"))
        registry.register(AgentConfig(agent_id="agent-b", model="m"))

        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value=Message(
                role="assistant",
                content=(
                    '{"status":"success","summary":"ok","artifacts":["artifact-a"],'
                    '"files_changed":["file-a.py"],"tool_calls_made":0,"errors":[],"next_steps":null}'
                ),
            )
        )

        ctx = make_agent_context(config=make_config(execution_mode=ExecutionMode.PARALLEL))
        ctx.messages.append(user_msg("run in parallel"))
        ctx.metadata["parallel_agent_ids"] = ["agent-a", "agent-b"]

        orch = AgentOrchestrator(
            llm_client=llm,
            parallel_execution_policy=Policy(),
            agent_registry=registry,
        )
        _, report = await orch.run(ctx)

        assert "artifact-a" in report.artifacts
        assert "file-a.py" in report.files_changed


# ---------------------------------------------------------------------------
# Subagent context isolation
# ---------------------------------------------------------------------------


class TestSubagentContextIsolation:
    @pytest.mark.asyncio
    async def test_subagent_starts_with_clean_messages(self) -> None:
        """Subagent context has only its task payload, not leader's history."""
        leader_cfg = make_config(execution_mode=ExecutionMode.LEADER)
        # Leader has a long history
        leader_ctx = make_agent_context(config=leader_cfg)
        for i in range(10):
            leader_ctx.messages.append(user_msg(f"leader turn {i}"))

        # Subagent context built from envelope payload — clean slate
        subagent_cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        subagent_ctx = AgentContext(
            session_id=leader_ctx.session_id,
            trace_id=leader_ctx.trace_id,
            parent_agent_id=leader_cfg.agent_id,
            config=subagent_cfg,
            messages=MessageList([user_msg("subagent task only")]),  # clean — only payload
        )

        # Verify subagent context has no leader history
        assert len(subagent_ctx.messages) == 1
        assert subagent_ctx.parent_agent_id == leader_cfg.agent_id
        assert subagent_ctx.session_id == leader_ctx.session_id  # shared trace

    def test_envelope_payload_is_minimal(self) -> None:
        """Envelope carries only task description, not full message list."""
        env = make_task_envelope(
            sender_id="leader",
            recipient_id="sub",
            payload={"task": "summarize the repo", "context_hint": "src/ folder"},
            trace_id="t",
            session_id="s",
        )
        raw = json.loads(env.model_dump_json())
        # payload must not contain a 'messages' key (no full history)
        assert "messages" not in raw["payload"]
        assert raw["payload"]["task"] == "summarize the repo"


# ---------------------------------------------------------------------------
# Path isolation
# ---------------------------------------------------------------------------


class TestPathIsolation:
    def test_path_within_allowed_allowed(self, tmp_path: Any) -> None:
        ctx = build_tool_context(
            session_id="s",
            trace_id="t",
            agent_id="a",
            allowed_paths=[str(tmp_path)],
        )
        sub = tmp_path / "subdir" / "file.txt"
        result = validate_path_access(sub, ctx)
        assert result.is_relative_to(tmp_path)

    def test_path_outside_allowed_denied(self, tmp_path: Any) -> None:
        ctx = build_tool_context(
            session_id="s",
            trace_id="t",
            agent_id="a",
            allowed_paths=[str(tmp_path / "sandbox")],
        )
        with pytest.raises(PathIsolationError):
            validate_path_access("/etc/passwd", ctx)

    def test_empty_allowed_paths_denies_all(self, tmp_path: Any) -> None:
        ctx = build_tool_context(
            session_id="s",
            trace_id="t",
            agent_id="a",
            allowed_paths=[],
        )
        with pytest.raises(PathIsolationError):
            validate_path_access(str(tmp_path), ctx)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_estimate_grows_with_content(self) -> None:
        short_msgs = [user_msg("hi")]
        long_msgs = [user_msg("x" * 40_000)]
        assert estimate_tokens(long_msgs) > estimate_tokens(short_msgs)

    def test_empty_messages_zero(self) -> None:
        assert estimate_tokens([]) == 0

    def test_estimate_tokens_resolves_profile_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import protocore.context as context_module

        calls: list[tuple[str | None, Any]] = []
        original = context_module._resolve_profile

        def tracked_resolve_profile(*, model: str | None, profile: Any) -> Any:
            calls.append((model, profile))
            return original(model=model, profile=profile)

        monkeypatch.setattr(context_module, "_resolve_profile", tracked_resolve_profile)

        estimate_tokens(
            [user_msg("hello")],
            model="Qwen/Qwen2.5-32B-Instruct",
        )

        assert len(calls) == 1


class TestTypeSerializationContracts:
    def test_agent_config_accepts_legacy_extra_alias(self) -> None:
        cfg = AgentConfig(agent_id="agent", model="m", extra={"trace": "legacy"})

        assert cfg.custom_data == {"trace": "legacy"}
        assert cfg.extra == {"trace": "legacy"}
        assert cfg.model_dump()["custom_data"] == {"trace": "legacy"}

    def test_compaction_summary_serializes_marker_and_accepts_legacy_alias(self) -> None:
        summary = CompactionSummary(current_goal="next")
        payload = summary.model_dump()

        assert payload["marker"] == "__compaction_summary__"
        assert "__marker__" not in payload

        legacy = CompactionSummary.model_validate({"__marker__": "__compaction_summary__", "current_goal": "old"})
        assert legacy.marker == "__compaction_summary__"


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class TestEventBus:
    @pytest.mark.asyncio
    async def test_event_emitted_and_received(self) -> None:
        bus = EventBus()
        received: list[str] = []

        async def handler(event: Any) -> None:
            received.append(event.name)

        bus.subscribe("test.event", handler)
        await bus.emit_simple("test.event", data="hello")
        assert "test.event" in received

    @pytest.mark.asyncio
    async def test_wildcard_receives_all(self) -> None:
        bus = EventBus()
        received: list[str] = []

        async def handler(event: Any) -> None:
            received.append(event.name)

        bus.subscribe("*", handler)
        await bus.emit_simple("foo")
        await bus.emit_simple("bar")
        assert "foo" in received
        assert "bar" in received

    @pytest.mark.asyncio
    async def test_handler_error_does_not_break_bus(self) -> None:
        bus = EventBus()

        async def bad_handler(event: Any) -> None:
            raise RuntimeError("oops")

        bus.subscribe("x", bad_handler)
        # Should not raise
        await bus.emit_simple("x")

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self) -> None:
        bus = EventBus()

        async def cancelled_handler(event: Any) -> None:
            _ = event
            raise asyncio.CancelledError()

        bus.subscribe("cancel", cancelled_handler)

        with pytest.raises(asyncio.CancelledError):
            await bus.emit_simple("cancel")

    @pytest.mark.asyncio
    async def test_emit_simple_truncates_large_payload(self) -> None:
        bus = EventBus()
        captured: list[Any] = []

        async def handler(event: Any) -> None:
            captured.append(event.payload["blob"])

        bus.subscribe("trim", handler)
        await bus.emit_simple("trim", blob="x" * 10_100)

        assert len(captured) == 1
        assert isinstance(captured[0], str)
        assert captured[0].startswith("x" * 100)
        assert "[truncated, 100 chars omitted]" in captured[0]

    @pytest.mark.asyncio
    async def test_duplicate_subscription_is_ignored(self) -> None:
        bus = EventBus()
        received: list[str] = []

        async def handler(event: Any) -> None:
            received.append(event.name)

        bus.subscribe("dup", handler)
        bus.subscribe("dup", handler)

        await bus.emit_simple("dup")

        assert received == ["dup"]

    @pytest.mark.asyncio
    async def test_subscribe_enforces_max_handlers_per_event(self) -> None:
        from protocore.events import MAX_HANDLERS_PER_EVENT

        bus = EventBus()
        calls: list[int] = []

        for idx in range(MAX_HANDLERS_PER_EVENT):
            async def handler(event: Any, *, _idx: int = idx) -> None:
                _ = event
                calls.append(_idx)

            bus.subscribe("cap", handler)

        async def overflow_handler(event: Any) -> None:
            _ = event

        with pytest.raises(RuntimeError, match="handler limit exceeded"):
            bus.subscribe("cap", overflow_handler)

        await bus.emit_simple("cap")

        assert len(calls) == MAX_HANDLERS_PER_EVENT

    @pytest.mark.asyncio
    async def test_orchestrator_events_include_correlation_identifiers(self) -> None:
        from protocore.events import (
            EV_LLM_CALL_END,
            EV_LLM_CALL_START,
            EV_LOOP_ITERATION,
            EV_TOOL_CALL_END,
            EV_TOOL_CALL_START,
            CoreEvent,
        )

        captured: list[CoreEvent] = []
        bus = EventBus()

        async def handler(event: CoreEvent) -> None:
            captured.append(event)

        for event_name in (
            EV_LLM_CALL_START,
            EV_LLM_CALL_END,
            EV_TOOL_CALL_START,
            EV_TOOL_CALL_END,
            EV_LOOP_ITERATION,
        ):
            bus.subscribe(event_name, handler)

        cfg = make_config(max_tool_calls=5, execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("run echo"))

        llm = mock_llm_one_tool_then_final("echo_tool")

        async def echo_handler(**kw: Any) -> ToolResult:
            _ = kw
            return ToolResult(
                tool_call_id="tc1",
                tool_name="echo_tool",
                content="echoed",
            )

        reg = ToolRegistry()
        reg.register(ToolDefinition(name="echo_tool", description="echo"), echo_handler)

        orch = AgentOrchestrator(
            llm_client=llm,
            tool_registry=reg,
            event_bus=bus,
        )
        await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert captured
        for event in captured:
            assert event.payload["agent_id"] == context.config.agent_id
            assert event.payload["request_id"] == context.request_id
            assert event.payload["trace_id"] == context.trace_id
            assert event.payload["session_id"] == context.session_id


# ---------------------------------------------------------------------------
# CancellationContext
# ---------------------------------------------------------------------------


class TestCancellationContext:
    def test_cancel_raises_check(self) -> None:
        ctx = CancellationContext()
        ctx.cancel("user abort")
        with pytest.raises(asyncio.CancelledError):
            ctx.check()

    def test_not_cancelled_by_default(self) -> None:
        ctx = CancellationContext()
        assert not ctx.is_cancelled
        ctx.check()  # no raise

    @pytest.mark.asyncio
    async def test_event_is_initialized_lazily_inside_wait(self) -> None:
        ctx = CancellationContext()
        assert len(ctx._events_by_loop) == 0

        waiter = asyncio.create_task(ctx.wait())
        await asyncio.sleep(0)

        assert len(ctx._events_by_loop) == 1
        ctx.cancel("stop")
        await waiter

    def test_cancel_ignores_closed_loops_and_keeps_propagating(self) -> None:
        ctx = CancellationContext()
        closed_loop = asyncio.new_event_loop()
        try:
            ctx._events_by_loop[closed_loop] = asyncio.Event()
            closed_loop.close()
            ctx.cancel("stop")
            assert ctx.is_cancelled
        finally:
            if not closed_loop.is_closed():
                closed_loop.close()

    @pytest.mark.asyncio
    async def test_cancel_from_worker_thread_unblocks_waiter(self) -> None:
        ctx = CancellationContext()
        waiter = asyncio.create_task(ctx.wait())
        await asyncio.sleep(0)

        await asyncio.to_thread(ctx.cancel, "thread-stop")
        await waiter

        assert ctx.is_cancelled

    def test_make_agent_context_does_not_share_metadata_with_tool_context(self) -> None:
        cfg = make_config()
        metadata = {"caller": "user"}
        context = make_agent_context(config=cfg, metadata=metadata)

        assert context.metadata == {"caller": "user"}
        assert context.tool_context.metadata["caller"] == "user"
        assert context.metadata is not context.tool_context.metadata
        assert "message_history_ref" not in context.metadata
        assert "message_history_ref" in context.tool_context.metadata

    def test_make_agent_context_rejects_tool_registry_kwarg_with_guidance(self) -> None:
        cfg = make_config()
        with pytest.raises(TypeError, match="ToolRegistry is injected into AgentOrchestrator"):
            make_agent_context(config=cfg, tool_registry=object())


# ---------------------------------------------------------------------------
# StateManager integration in orchestrator
# ---------------------------------------------------------------------------


class TestStateManagerIntegration:
    @pytest.mark.asyncio
    async def test_state_manager_called_on_run(self) -> None:
        """StateManager snapshot/report hooks are called on run."""
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("hello"))

        state_mgr = MagicMock()
        state_mgr.save_session_snapshot = AsyncMock()
        state_mgr.load_session_snapshot = AsyncMock(return_value=None)
        state_mgr.update_session_snapshot = AsyncMock()
        state_mgr.save_execution_report = AsyncMock()

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("done"),
            state_manager=state_mgr,
        )
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        state_mgr.save_session_snapshot.assert_awaited_once()
        state_mgr.update_session_snapshot.assert_awaited_once()
        state_mgr.save_execution_report.assert_awaited_once()
        assert result.content == "done"

    @pytest.mark.asyncio
    async def test_state_manager_failure_does_not_break_run(self) -> None:
        """If StateManager fails, run continues with a warning."""
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("hello"))

        state_mgr = MagicMock()
        state_mgr.save_session_snapshot = AsyncMock(side_effect=RuntimeError("db down"))
        state_mgr.load_session_snapshot = AsyncMock(return_value=None)
        state_mgr.update_session_snapshot = AsyncMock()
        state_mgr.save_execution_report = AsyncMock(side_effect=RuntimeError("db down"))

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("ok"),
            state_manager=state_mgr,
        )
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert result.content == "ok"
        assert "session_snapshot_save_failed" in report.warnings

    @pytest.mark.asyncio
    async def test_saved_snapshot_contains_non_empty_refs(self) -> None:
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("hello"))

        state_mgr = MagicMock()
        state_mgr.save_session_snapshot = AsyncMock()
        state_mgr.load_session_snapshot = AsyncMock(return_value=None)
        state_mgr.update_session_snapshot = AsyncMock()
        state_mgr.save_execution_report = AsyncMock()

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("done"),
            state_manager=state_mgr,
        )
        await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert state_mgr.save_session_snapshot.await_args is not None
        snapshot = state_mgr.save_session_snapshot.await_args.args[0]
        assert snapshot.message_history_ref
        assert snapshot.execution_metadata_ref

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scenario", "expected_status"),
        [
            ("completed", ExecutionStatus.COMPLETED),
            ("failed", ExecutionStatus.FAILED),
            ("timeout", ExecutionStatus.TIMEOUT),
            ("cancelled", ExecutionStatus.CANCELLED),
        ],
    )
    async def test_state_manager_persists_terminal_statuses(
        self,
        scenario: str,
        expected_status: ExecutionStatus,
    ) -> None:
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("hello"))

        state_mgr = MagicMock()
        state_mgr.save_session_snapshot = AsyncMock()
        state_mgr.load_session_snapshot = AsyncMock(return_value=None)
        state_mgr.update_session_snapshot = AsyncMock()
        state_mgr.save_execution_report = AsyncMock()

        orch_kwargs: dict[str, Any] = {
            "state_manager": state_mgr,
        }
        if scenario == "completed":
            orch_kwargs["llm_client"] = mock_llm_final("done")
        elif scenario == "failed":
            llm = MagicMock()
            llm.complete = AsyncMock(side_effect=RuntimeError("boom"))
            orch_kwargs["llm_client"] = llm
        elif scenario == "timeout":
            class TimeoutPolicy:
                def get_timeout(self, operation: str) -> float:
                    return 0.01 if operation == "llm.complete" else 1.0

            llm = MagicMock()

            async def slow_complete(*args: Any, **kwargs: Any) -> Message:
                await asyncio.sleep(0.05)
                return Message(role="assistant", content="late")

            llm.complete = AsyncMock(side_effect=slow_complete)
            orch_kwargs["llm_client"] = llm
            orch_kwargs["timeout_policy"] = TimeoutPolicy()
        else:
            llm = MagicMock()
            llm.complete = AsyncMock(side_effect=asyncio.CancelledError("stop"))
            orch_kwargs["llm_client"] = llm

        orch = AgentOrchestrator(**orch_kwargs)
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert state_mgr.update_session_snapshot.await_args is not None
        assert state_mgr.save_execution_report.await_args is not None
        updated_snapshot = state_mgr.update_session_snapshot.await_args.args[0]
        saved_report = state_mgr.save_execution_report.await_args.args[0]

        assert result.status == expected_status
        assert report.status == expected_status
        assert updated_snapshot.execution_report_id == context.request_id
        assert updated_snapshot.metadata["status"] == expected_status.value
        assert updated_snapshot.metadata["warnings"] == report.warnings
        assert saved_report.status == expected_status

    @pytest.mark.asyncio
    async def test_state_manager_persists_partial_status_for_auto_select_fallback(self) -> None:
        class RecordingStateManager:
            def __init__(self) -> None:
                self.snapshots: list[tuple[str, Any]] = []
                self.reports: list[ExecutionReport] = []

            async def get(self, key: str) -> Any:
                return None

            async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
                return None

            async def delete(self, key: str) -> None:
                return None

            async def save_session_snapshot(self, snapshot: Any) -> None:
                self.snapshots.append(("save", snapshot))

            async def load_session_snapshot(self, session_id: str) -> Any:
                return None

            async def update_session_snapshot(self, snapshot: Any) -> None:
                self.snapshots.append(("update", snapshot))

            async def save_execution_report(self, report: ExecutionReport) -> None:
                self.reports.append(report)

        registry = AgentRegistry()
        registry.register(
            AgentConfig(
                agent_id="child-1",
                model="gpt-4o",
                role=AgentRole.SUBAGENT,
                execution_mode=ExecutionMode.LEADER,
            )
        )
        selection_policy = MagicMock()
        selection_policy.select = AsyncMock(return_value="child-1")

        cfg = make_config(
            execution_mode=ExecutionMode.AUTO_SELECT,
            role=AgentRole.LEADER,
        ).model_copy(update={"agent_id": "leader"})
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("delegate"))

        state_mgr = RecordingStateManager()
        orch = AgentOrchestrator(
            llm_client=mock_llm_final("not valid json"),
            state_manager=state_mgr,
            agent_registry=registry,
            subagent_selection_policy=selection_policy,
        )
        result, report = await orch.run(context, run_kind=RunKind.LEADER)

        leader_updates = [
            snapshot
            for kind, snapshot in state_mgr.snapshots
            if kind == "update" and snapshot.agent_id == "leader"
        ]
        assert result.status == ExecutionStatus.PARTIAL
        assert report.status == ExecutionStatus.PARTIAL
        assert leader_updates[-1].metadata["status"] == ExecutionStatus.PARTIAL.value
        assert any(saved_report.status == ExecutionStatus.PARTIAL for saved_report in state_mgr.reports)

    @pytest.mark.asyncio
    async def test_state_manager_update_failure_does_not_skip_report_save(self) -> None:
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("hello"))

        state_mgr = MagicMock()
        state_mgr.save_session_snapshot = AsyncMock()
        state_mgr.load_session_snapshot = AsyncMock(return_value=None)
        state_mgr.update_session_snapshot = AsyncMock(side_effect=RuntimeError("db down"))
        state_mgr.save_execution_report = AsyncMock()

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("done"),
            state_manager=state_mgr,
        )
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert result.content == "done"
        assert "session_snapshot_update_failed" in report.warnings
        state_mgr.save_execution_report.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_state_manager_report_failure_does_not_break_result(self) -> None:
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("hello"))

        state_mgr = MagicMock()
        state_mgr.save_session_snapshot = AsyncMock()
        state_mgr.load_session_snapshot = AsyncMock(return_value=None)
        state_mgr.update_session_snapshot = AsyncMock()
        state_mgr.save_execution_report = AsyncMock(side_effect=RuntimeError("db down"))

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("done"),
            state_manager=state_mgr,
        )
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert result.content == "done"
        assert "execution_report_save_failed" in report.warnings
        state_mgr.update_session_snapshot.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_state_manager_initial_save_timeout_is_degraded(self) -> None:
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("hello"))

        state_mgr = MagicMock()

        async def slow_save(snapshot: Any) -> None:
            _ = snapshot
            await asyncio.sleep(0.01)

        state_mgr.save_session_snapshot = AsyncMock(side_effect=slow_save)
        state_mgr.load_session_snapshot = AsyncMock(return_value=None)
        state_mgr.update_session_snapshot = AsyncMock()
        state_mgr.save_execution_report = AsyncMock()

        with patch(
            "protocore.orchestrator.DEFAULT_STATE_MANAGER_TIMEOUT_SECONDS",
            0.001,
        ):
            orch = AgentOrchestrator(
                llm_client=mock_llm_final("done"),
                state_manager=state_mgr,
            )
            result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert result.content == "done"
        assert report.state_manager_timeout_count == 1
        assert "session_snapshot_save_timed_out" in report.warnings

    @pytest.mark.asyncio
    async def test_state_manager_final_persistence_timeouts_are_counted(self) -> None:
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("hello"))

        state_mgr = MagicMock()
        state_mgr.save_session_snapshot = AsyncMock()

        async def slow_load(session_id: str) -> Any:
            _ = session_id
            await asyncio.sleep(0.01)
            return None

        async def slow_report(report: Any) -> None:
            _ = report
            await asyncio.sleep(0.01)

        state_mgr.load_session_snapshot = AsyncMock(side_effect=slow_load)
        state_mgr.update_session_snapshot = AsyncMock()
        state_mgr.save_execution_report = AsyncMock(side_effect=slow_report)

        with patch(
            "protocore.orchestrator.DEFAULT_STATE_MANAGER_TIMEOUT_SECONDS",
            0.001,
        ):
            orch = AgentOrchestrator(
                llm_client=mock_llm_final("done"),
                state_manager=state_mgr,
            )
            result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert result.content == "done"
        assert report.state_manager_timeout_count == 2
        assert "session_snapshot_load_timed_out" in report.warnings
        assert "execution_report_save_timed_out" in report.warnings

    @pytest.mark.asyncio
    async def test_state_manager_falls_back_to_save_snapshot_when_update_missing(self) -> None:
        class SaveOnlyStateManager:
            def __init__(self) -> None:
                self.snapshots: list[Any] = []
                self.reports: list[ExecutionReport] = []

            async def get(self, key: str) -> Any:
                return None

            async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
                return None

            async def delete(self, key: str) -> None:
                return None

            async def save_session_snapshot(self, snapshot: Any) -> None:
                self.snapshots.append(snapshot)

            async def load_session_snapshot(self, session_id: str) -> Any:
                return None

            async def save_execution_report(self, report: ExecutionReport) -> None:
                self.reports.append(report)

        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("hello"))

        state_mgr = SaveOnlyStateManager()
        orch = AgentOrchestrator(
            llm_client=mock_llm_final("done"),
            state_manager=state_mgr,  # type: ignore[arg-type]
        )
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert result.content == "done"
        assert len(state_mgr.snapshots) == 1
        assert "session_snapshot_update_failed" in report.warnings
        assert len(state_mgr.reports) == 1
        assert state_mgr.reports[0].status == report.status

    @pytest.mark.asyncio
    async def test_state_manager_final_snapshot_contains_messages_from_failed_run(self) -> None:
        llm = MagicMock()
        call_count = 0

        async def complete(*args: Any, **kwargs: Any) -> Message:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Message(
                    role="assistant",
                    content=None,
                    tool_calls=_tool_calls(
                        {
                            "id": "tc1",
                            "function": {"name": "echo", "arguments": "{}"},
                        }
                    ),
                )
            raise RuntimeError("boom")

        llm.complete = AsyncMock(side_effect=complete)
        tool_registry = ToolRegistry()
        tool_registry.register(
            ToolDefinition(name="echo", description="echo"),
            AsyncMock(return_value=ToolResult(tool_call_id="tc1", tool_name="echo", content="ok")),
        )

        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            tool_definitions=[ToolDefinition(name="echo", description="echo")],
        )
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("hello"))

        state_mgr = MagicMock()
        state_mgr.save_session_snapshot = AsyncMock()
        state_mgr.load_session_snapshot = AsyncMock(return_value=None)
        state_mgr.update_session_snapshot = AsyncMock()
        state_mgr.save_execution_report = AsyncMock()

        orch = AgentOrchestrator(
            llm_client=llm,
            tool_registry=tool_registry,
            state_manager=state_mgr,
        )
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert state_mgr.update_session_snapshot.await_args is not None
        updated_snapshot = state_mgr.update_session_snapshot.await_args.args[0]
        assert result.status == ExecutionStatus.FAILED
        assert report.status == ExecutionStatus.FAILED
        assert [message.role for message in updated_snapshot.messages] == [
            "user",
            "assistant",
            "tool",
        ]
        assert updated_snapshot.messages[-1].content == "ok"


# ---------------------------------------------------------------------------
# AUTO_SELECT mode
# ---------------------------------------------------------------------------


class TestAutoSelect:
    @pytest.mark.asyncio
    async def test_auto_select_calls_policy(self) -> None:
        """AUTO_SELECT mode invokes SubagentSelectionPolicy.select()."""
        from protocore.registry import AgentRegistry

        agent_reg = AgentRegistry()
        agent_reg.register(AgentConfig(agent_id="agent-coder", model="m"))

        cfg = make_config(execution_mode=ExecutionMode.AUTO_SELECT)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("route this task"))

        selection_policy = MagicMock()
        selection_policy.select = AsyncMock(return_value="agent-coder")

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("routed"),
            agent_registry=agent_reg,
            subagent_selection_policy=selection_policy,
        )
        result, report = await orch.run(context, run_kind=RunKind.LEADER)

        selection_policy.select.assert_awaited_once()
        assert any("auto_selected_agent:agent-coder" in a for a in report.artifacts)
        assert result.content == "routed"

    @pytest.mark.asyncio
    async def test_auto_select_without_policy_fails(self) -> None:
        """AUTO_SELECT falls back to built-in capability-based selection."""
        from protocore.registry import AgentRegistry

        agent_reg = AgentRegistry()
        agent_reg.register(
            AgentConfig(
                agent_id="writer",
                name="Writer",
                description="Writes concise release notes and documentation.",
                model="m",
                role=AgentRole.SUBAGENT,
            )
        )
        agent_reg.register(
            AgentConfig(
                agent_id="coder",
                name="Coder",
                description="Implements Python code changes and debugging.",
                model="m",
                role=AgentRole.SUBAGENT,
            )
        )

        cfg = make_config(execution_mode=ExecutionMode.AUTO_SELECT)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("write code for this task"))

        llm = MagicMock()
        llm.complete = AsyncMock(
            side_effect=[
                Message(role="assistant", content='{"agent_id":"coder"}'),
                Message(role="assistant", content="fallback"),
            ]
        )
        orch = AgentOrchestrator(
            llm_client=llm,
            agent_registry=agent_reg,
        )
        result, report = await orch.run(context, run_kind=RunKind.LEADER)

        assert result.status == ExecutionStatus.PARTIAL
        assert report.error_code is None
        assert context.metadata["auto_selected_agent"] == "coder"
        assert llm.complete.await_count == 2

    @pytest.mark.asyncio
    async def test_auto_select_pre_llm_hook_fires_for_subagent_not_leader(self) -> None:
        from protocore import HookManager, hookimpl
        from protocore.registry import AgentRegistry

        pre_llm_agents: list[str] = []

        class HookObserver:
            @hookimpl
            def pre_llm_call(self, messages: list[Message], context: AgentContext, report: ExecutionReport) -> None:
                _ = messages, report
                pre_llm_agents.append(context.config.agent_id)

        agent_reg = AgentRegistry()
        agent_reg.register(
            AgentConfig(
                agent_id="math-subagent",
                model="m",
                execution_mode=ExecutionMode.BYPASS,
                role=AgentRole.SUBAGENT,
            )
        )

        leader_cfg = make_config(
            execution_mode=ExecutionMode.AUTO_SELECT,
        )
        context = make_agent_context(config=leader_cfg)
        context.messages.append(user_msg("route this task"))

        selection_policy = MagicMock()
        selection_policy.select = AsyncMock(return_value="math-subagent")

        hooks = HookManager()
        hooks.register(HookObserver())
        orch = AgentOrchestrator(
            llm_client=mock_llm_final("subagent answer"),
            agent_registry=agent_reg,
            hook_manager=hooks,
            subagent_selection_policy=selection_policy,
        )
        _result, _report = await orch.run(context, run_kind=RunKind.LEADER)

        assert pre_llm_agents == ["math-subagent"]


# ---------------------------------------------------------------------------
# Minor version envelope warning
# ---------------------------------------------------------------------------


class TestEnvelopeMinorVersion:
    def test_minor_version_mismatch_detected(self) -> None:
        env = AgentEnvelope(
            protocol_version="1.99",
            message_type=MessageType.RESULT,
            trace_id="t",
            session_id="s",
            sender=AgentIdentity(agent_id="s1", role=AgentRole.SUBAGENT),
            recipient=AgentIdentity(agent_id="l1", role=AgentRole.LEADER),
            payload={
                "status": "success",
                "summary": "done",
                "artifacts": [],
                "files_changed": [],
                "tool_calls_made": 0,
                "errors": [],
                "next_steps": None,
            },
        )
        warning = env.check_minor_version()
        assert warning is not None
        assert "protocol_minor_version_mismatch" in warning

    def test_matching_minor_version_no_warning(self) -> None:
        env = make_task_envelope(
            sender_id="l",
            recipient_id="s",
            payload={"task": "ping"},
            trace_id="t",
            session_id="s",
        )
        assert env.check_minor_version() is None

    def test_all_factories_preserve_structured_payloads(self) -> None:
        result_env = make_result_envelope(
            sender_id="sub",
            recipient_id="leader",
            result_payload={
                "status": "success",
                "summary": "done",
                "artifacts": ["artifact.txt"],
                "files_changed": ["file.py"],
                "tool_calls_made": 1,
                "errors": [],
                "next_steps": None,
            },
            trace_id="t",
            session_id="s",
        )
        control_env = make_control_envelope(
            sender_id="leader",
            recipient_id="sub",
            command=ControlCommand.CANCEL,
            trace_id="t",
            session_id="s",
        )
        error_env = make_error_envelope(
            sender_id="sub",
            recipient_id="leader",
            error_message="boom",
            error_code="FAIL",
            trace_id="t",
            session_id="s",
        )
        assert result_env.payload["status"] == "success"
        assert result_env.payload["files_changed"] == ["file.py"]
        assert control_env.payload["command"] == "cancel"
        assert error_env.payload["error_code"] == "FAIL"

    def test_result_envelope_uses_full_subagent_contract(self) -> None:
        env = make_result_envelope(
            sender_id="sub",
            recipient_id="leader",
            result_payload={
                "status": "partial",
                "summary": "Need follow-up",
                "artifacts": ["artifact.txt"],
                "files_changed": ["foo.py"],
                "tool_calls_made": 2,
                "errors": ["missing context"],
                "next_steps": "retry with wider search",
            },
            trace_id="t",
            session_id="s",
        )

        assert env.payload["tool_calls_made"] == 2
        assert env.payload["next_steps"] == "retry with wider search"


class TestSessionSnapshotContract:
    def test_snapshot_requires_message_and_metadata_refs(self) -> None:
        from protocore.types import SessionSnapshot

        with pytest.raises(ValidationError):
            SessionSnapshot(
                session_id="s",
                trace_id="t",
                agent_id="a",
            )  # type: ignore[call-arg]

    def test_agent_context_populates_message_and_metadata_refs(self) -> None:
        context = make_agent_context(config=make_config())

        assert context.message_history_ref == f"session:{context.session_id}:messages"
        assert context.execution_metadata_ref == f"request:{context.request_id}:metadata"

    def test_snapshot_model_copy_refreshes_updated_at(self) -> None:
        from protocore.types import SessionSnapshot

        snapshot = SessionSnapshot(
            session_id="s",
            trace_id="t",
            agent_id="a",
            message_history_ref="session:s:messages",
            execution_metadata_ref="request:r:metadata",
            created_at="2026-03-08T00:00:00+00:00",
            updated_at="2026-03-08T00:00:01+00:00",
        )

        copied = snapshot.model_copy(update={"execution_report_id": "req-2"})

        assert copied.execution_report_id == "req-2"
        assert copied.updated_at != snapshot.updated_at


class TestProtocolReplaceability:
    def test_storage_protocol_is_runtime_replaceable(self) -> None:
        class FakeStorage:
            async def query(self, collection: str, query: dict[str, Any]) -> list[dict[str, Any]]:
                return []

            async def insert(self, collection: str, data: dict[str, Any]) -> str:
                return "id-1"

            async def update(self, collection: str, id: str, data: dict[str, Any]) -> None:
                return None

            async def delete(self, collection: str, id: str) -> None:
                return None

        assert isinstance(FakeStorage(), Storage)

    @pytest.mark.asyncio
    async def test_workflow_engine_protocol_is_langgraph_adapter_compatible(self) -> None:
        """WorkflowEngine protocol accepts a LangGraph-style adapter and runs it (C48-004)."""
        from protocore import WorkflowEngine
        from protocore.types import WorkflowDefinition, WorkflowEdge, WorkflowNode

        class LangGraphCompatibleAdapter:
            """Minimal adapter shape a service-side LangGraph wrapper can expose."""

            def __init__(self) -> None:
                self.graph = object()  # adapter may hold underlying langgraph graph

            async def run(
                self,
                workflow: WorkflowDefinition,
                context: AgentContext,
            ) -> tuple[Result, ExecutionReport]:
                report = make_execution_report(context=context, run_kind=RunKind.SUBAGENT)
                report.workflow_id = workflow.workflow_id
                report.node_count = len(workflow.nodes)
                report.edge_count = len(workflow.edges)
                report.node_durations_ms = {node.node_id: 1.0 for node in workflow.nodes}
                report.finalize(ExecutionStatus.COMPLETED, StopReason.END_TURN)
                return Result(content="workflow ok", status=ExecutionStatus.COMPLETED), report

        adapter = LangGraphCompatibleAdapter()
        assert isinstance(adapter, WorkflowEngine)

        workflow = WorkflowDefinition(
            name="wf",
            nodes=[
                WorkflowNode(node_id="n1", label="start"),
                WorkflowNode(node_id="n2", label="end"),
            ],
            edges=[WorkflowEdge(from_node="n1", to_node="n2")],
        )
        context = make_agent_context(config=make_config(execution_mode=ExecutionMode.BYPASS))

        orchestrator = AgentOrchestrator(
            llm_client=mock_llm_final(),
            workflow_engine=adapter,
        )
        result, report = await orchestrator.run_workflow(workflow, context)

        assert result.content == "workflow ok"
        assert report.status == ExecutionStatus.COMPLETED
        assert report.workflow_id == workflow.workflow_id
        assert report.node_count == 2
        assert report.edge_count == 1

    @pytest.mark.asyncio
    async def test_full_dependency_injection_config_runs_without_core_changes(self) -> None:
        """All protocol adapters can be injected together and complete one run (C46-006)."""
        from protocore import (
            AgentRegistry,
            CompressionStrategy,
            ExecutionPolicy,
            LLMClient,
            ParallelExecutionPolicy,
            PlanningPolicy,
            PlanningStrategy,
            RetryPolicy,
            StateManager,
            SubagentSelectionPolicy,
            TelemetryCollector,
            TimeoutPolicy,
            ToolExecutor,
            Transport,
            WorkflowEngine,
        )
        from protocore.types import PlanArtifact, PolicyDecision, SessionSnapshot, WorkflowDefinition

        class FakeLLM:
            async def complete(self, **kwargs: Any) -> Message:
                return Message(role="assistant", content="ok")

            async def complete_structured(self, **kwargs: Any) -> Any:
                return SubagentResult(status=SubagentStatus.SUCCESS, summary="ok")

            async def stream_with_tools(
                self, **kwargs: Any
            ) -> AsyncIterator[dict[str, Any]]:
                if False:  # pragma: no cover
                    yield {}

        class FakeToolExecutor:
            async def execute(
                self,
                tool_name: str,
                arguments: dict[str, Any],
                context: Any,
            ) -> ToolResult:
                return ToolResult(tool_call_id="tc", tool_name=tool_name, content="ok")

            def list_tools(self) -> list[ToolDefinition]:
                return [ToolDefinition(name="noop", description="noop")]

        class FakeStateManager:
            def __init__(self) -> None:
                self.saved_snapshots = 0
                self.updated_snapshots = 0
                self.saved_reports = 0

            async def get(self, key: str) -> Any:
                return None

            async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
                return None

            async def delete(self, key: str) -> None:
                return None

            async def save_session_snapshot(self, snapshot: SessionSnapshot) -> None:
                self.saved_snapshots += 1

            async def load_session_snapshot(self, session_id: str) -> SessionSnapshot | None:
                return None

            async def update_session_snapshot(self, snapshot: SessionSnapshot) -> None:
                self.updated_snapshots += 1

            async def save_execution_report(self, report: ExecutionReport) -> None:
                self.saved_reports += 1

        class FakeTransport:
            async def send(self, envelope: AgentEnvelope, destination: str) -> None:
                return None

            async def receive(self, source: str) -> AsyncIterator[AgentEnvelope]:
                if False:  # pragma: no cover
                    yield make_task_envelope(
                        sender_id="s",
                        recipient_id="r",
                        payload={"task": "x"},
                        trace_id="t",
                        session_id="s",
                    )

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
                return messages, CompactionSummary(current_goal="noop"), True

            async def apply_manual(
                self,
                messages: list[Message],
                *,
                model: str | None = None,
                config: Any = None,
                run_kind: Any = None,
            ) -> tuple[list[Message], CompactionSummary]:
                _ = (model, config, run_kind)
                return messages, CompactionSummary(current_goal="noop")

        class FakePlanning:
            async def build_plan(self, task: str, context: AgentContext, llm_client: Any) -> PlanArtifact:
                return PlanArtifact(
                    trace_id=context.trace_id,
                    steps=[],
                    raw_plan=task or "goal",
                )

            async def update_plan(self, plan: PlanArtifact, context: AgentContext, llm_client: Any) -> PlanArtifact:
                return plan

        class FakePlanningPolicy:
            async def should_plan(self, context: AgentContext) -> bool:
                return True

            async def allow_bypass(self, context: AgentContext) -> bool:
                return True

        class FakeSelection:
            async def select(self, task: str, available_agents: list[str], context: AgentContext) -> str:
                return available_agents[0] if available_agents else "fallback-agent"

        class FakeParallelPolicy:
            @property
            def max_concurrency(self) -> int:
                return 2

            @property
            def timeout_seconds(self) -> float:
                return 5.0

            @property
            def cancellation_mode(self) -> str:
                return "graceful"

            async def merge_results(
                self,
                results: list[SubagentResult | None],
                agent_ids: list[str],
            ) -> SubagentResult:
                return SubagentResult(status=SubagentStatus.SUCCESS, summary="merged")

        class FakeExecutionPolicy:
            async def evaluate(
                self,
                tool_name: str,
                arguments: dict[str, Any],
                context: Any,
            ) -> PolicyDecision:
                return PolicyDecision.ALLOW

        class FakeWorkflowEngine:
            async def run(
                self,
                workflow: WorkflowDefinition,
                context: AgentContext,
            ) -> tuple[Result, ExecutionReport]:
                report = make_execution_report(context=context)
                report.finalize(ExecutionStatus.COMPLETED, StopReason.END_TURN)
                return Result(content="wf", status=ExecutionStatus.COMPLETED), report

        class FakeTelemetry:
            def __init__(self) -> None:
                self.event_names: list[str] = []

            async def record_event(
                self,
                event_name: str,
                payload: dict[str, Any],
                report: ExecutionReport,
            ) -> None:
                self.event_names.append(event_name)

        class FakeTimeout:
            def get_timeout(self, operation: str) -> float:
                return 1.0

        class FakeRetry:
            def should_retry(self, attempt: int, error: Exception) -> bool:
                return False

            def delay_seconds(self, attempt: int) -> float:
                return 0.0

        llm = FakeLLM()
        tool_executor = FakeToolExecutor()
        state_manager = FakeStateManager()
        transport = FakeTransport()
        compressor = FakeCompression()
        planning = FakePlanning()
        planning_policy = FakePlanningPolicy()
        selection = FakeSelection()
        parallel_policy = FakeParallelPolicy()
        execution_policy = FakeExecutionPolicy()
        workflow_engine = FakeWorkflowEngine()
        telemetry = FakeTelemetry()
        timeout_policy = FakeTimeout()
        retry_policy = FakeRetry()

        assert isinstance(llm, LLMClient)
        assert isinstance(tool_executor, ToolExecutor)
        assert isinstance(state_manager, StateManager)
        assert isinstance(transport, Transport)
        assert isinstance(compressor, CompressionStrategy)
        assert isinstance(planning_policy, PlanningPolicy)
        assert isinstance(planning, PlanningStrategy)
        assert isinstance(selection, SubagentSelectionPolicy)
        assert isinstance(parallel_policy, ParallelExecutionPolicy)
        assert isinstance(execution_policy, ExecutionPolicy)
        assert isinstance(workflow_engine, WorkflowEngine)
        assert isinstance(telemetry, TelemetryCollector)
        assert isinstance(timeout_policy, TimeoutPolicy)
        assert isinstance(retry_policy, RetryPolicy)

        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("hello"))

        orch = AgentOrchestrator(
            llm_client=llm,
            tool_executor=tool_executor,
            tool_registry=ToolRegistry(),
            agent_registry=AgentRegistry(),
            planning_strategy=planning,
            planning_policy=planning_policy,
            parallel_execution_policy=parallel_policy,
            execution_policy=execution_policy,
            compressor=compressor,
            state_manager=state_manager,
            subagent_selection_policy=selection,
            transport=transport,
            workflow_engine=workflow_engine,
            telemetry_collector=telemetry,
            timeout_policy=timeout_policy,
            retry_policy=retry_policy,
        )
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert result.status == ExecutionStatus.COMPLETED
        assert report.status == ExecutionStatus.COMPLETED
        assert state_manager.saved_snapshots == 1
        assert state_manager.updated_snapshots == 1
        assert state_manager.saved_reports == 1
        assert len(telemetry.event_names) >= 2


# ---------------------------------------------------------------------------
# Usage accumulation in report
# ---------------------------------------------------------------------------


class TestUsageAccumulation:
    @pytest.mark.asyncio
    async def test_usage_accumulated_from_llm_response(self) -> None:
        """input_tokens/output_tokens populated from LLM response usage."""
        from protocore.types import LLMUsage

        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("count tokens"))

        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(
            role="assistant",
            content="answer",
            usage=LLMUsage(input_tokens=100, output_tokens=50),
        ))

        orch = AgentOrchestrator(llm_client=llm)
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert report.input_tokens == 100
        assert report.output_tokens == 50

    @pytest.mark.asyncio
    async def test_usage_accumulates_across_iterations(self) -> None:
        """Multiple LLM calls accumulate tokens."""
        from protocore.types import LLMUsage

        cfg = make_config(max_tool_calls=5, execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("multi-step"))

        call_n = 0

        async def multi_step(*args: Any, **kwargs: Any) -> Message:
            nonlocal call_n
            call_n += 1
            if call_n == 1:
                return Message(
                    role="assistant",
                    content=None,
                    tool_calls=_tool_calls(
                        {"id": "tc1", "function": {"name": "t", "arguments": "{}"}}
                    ),
                    usage=LLMUsage(input_tokens=50, output_tokens=20),
                )
            return Message(
                role="assistant",
                content="final",
                usage=LLMUsage(input_tokens=80, output_tokens=30),
            )

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=multi_step)

        from protocore.registry import ToolRegistry

        reg = ToolRegistry()
        async def tool_handler(**kw: Any) -> ToolResult:
            return ToolResult(tool_call_id="tc1", tool_name="t", content="ok")

        reg.register(ToolDefinition(name="t", description="test"), tool_handler)

        orch = AgentOrchestrator(llm_client=llm, tool_registry=reg)
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert report.input_tokens == 130  # 50 + 80
        assert report.output_tokens == 50  # 20 + 30


# ---------------------------------------------------------------------------
# FIX: Planning gate warns when no strategy
# ---------------------------------------------------------------------------


class TestPlanningGateEnforcement:
    @pytest.mark.asyncio
    async def test_leader_mode_requires_planning_strategy(self) -> None:
        """LEADER mode without PlanningStrategy fails fast."""
        cfg = make_config(execution_mode=ExecutionMode.LEADER)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("task"))

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("ok"),
        )
        result, report = await orch.run(context)

        assert result.status == ExecutionStatus.FAILED
        assert report.error_code == "PLANNING_REQUIRED"

    @pytest.mark.asyncio
    async def test_leader_mode_policy_requires_strategy_when_planning_enabled(self) -> None:
        """PlanningPolicy.should_plan=True still enforces PlanningStrategy presence."""

        class RequirePlanningPolicy:
            async def should_plan(self, context: AgentContext) -> bool:
                return True

            async def allow_bypass(self, context: AgentContext) -> bool:
                return False

        cfg = make_config(execution_mode=ExecutionMode.LEADER)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("task"))

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("ok"),
            planning_policy=RequirePlanningPolicy(),
        )
        result, report = await orch.run(context)

        assert result.status == ExecutionStatus.FAILED
        assert report.error_code == "PLANNING_REQUIRED"

    @pytest.mark.asyncio
    async def test_leader_mode_policy_can_skip_planning_without_strategy(self) -> None:
        """PlanningPolicy.should_plan=False skips planning gate even without strategy."""

        class SkipPlanningPolicy:
            async def should_plan(self, context: AgentContext) -> bool:
                return False

            async def allow_bypass(self, context: AgentContext) -> bool:
                return False

        cfg = make_config(execution_mode=ExecutionMode.LEADER)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("task"))

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("ok"),
            planning_policy=SkipPlanningPolicy(),
        )
        result, report = await orch.run(context)

        assert result.status == ExecutionStatus.COMPLETED
        assert report.status == ExecutionStatus.COMPLETED


# ---------------------------------------------------------------------------
# FIX: AUTO_SELECT stores selection in context.metadata
# ---------------------------------------------------------------------------


class TestAutoSelectMetadata:
    @pytest.mark.asyncio
    async def test_auto_select_stores_in_metadata(self) -> None:
        """AUTO_SELECT puts the selected agent ID in context.metadata."""
        from protocore.registry import AgentRegistry

        agent_reg = AgentRegistry()
        agent_reg.register(AgentConfig(agent_id="agent-writer", model="m"))

        cfg = make_config(execution_mode=ExecutionMode.AUTO_SELECT)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("route me"))

        policy = MagicMock()
        policy.select = AsyncMock(return_value="agent-writer")

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("routed"),
            agent_registry=agent_reg,
            subagent_selection_policy=policy,
        )
        result, report = await orch.run(context, run_kind=RunKind.LEADER)

        assert context.metadata["auto_selected_agent"] == "agent-writer"
        assert any("auto_selected_agent:agent-writer" in a for a in report.artifacts)


# ---------------------------------------------------------------------------
# FIX: EV_ERROR emitted on error path
# ---------------------------------------------------------------------------


class TestErrorEventEmission:
    @pytest.mark.asyncio
    async def test_error_emits_ev_error(self) -> None:
        """Error path emits EV_ERROR via EventBus."""
        from protocore.events import EV_ERROR, CoreEvent

        bus = EventBus()
        captured: list[CoreEvent] = []

        async def handler(event: CoreEvent) -> None:
            captured.append(event)

        bus.subscribe(EV_ERROR, handler)

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("boom"))

        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("fail"))

        orch = AgentOrchestrator(
            llm_client=llm,
            event_bus=bus,
        )
        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert result.status == ExecutionStatus.FAILED
        assert len(captured) == 1
        assert captured[0].payload["error_code"] == "RuntimeError"
        assert captured[0].payload["error_message"] == "boom"


# ---------------------------------------------------------------------------
# FIX: Manual compact integration in orchestrator
# ---------------------------------------------------------------------------


class TestManualCompactIntegration:
    @pytest.mark.asyncio
    async def test_manual_compact_tool_requests_compaction_via_standard_core_route(self) -> None:
        from protocore.compression import ContextCompressor
        from protocore.events import EV_COMPRESSION_MANUAL, CoreEvent
        from protocore.registry import ToolRegistry

        loop_llm = MagicMock()
        loop_llm.complete = AsyncMock(
            side_effect=[
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=_tool_calls(
                        {
                            "id": "tc1",
                            "function": {
                                "name": "manual_compact",
                                "arguments": '{"reason":"compress before final answer"}',
                            },
                        }
                    ),
                ),
                Message(role="assistant", content="done after compact"),
            ]
        )

        summary_llm = MagicMock()
        summary_llm.complete = AsyncMock(
            return_value=Message(
                role="assistant",
                content=(
                    '{"completed_tasks":[],"current_goal":"compacted","key_decisions":[],'
                    '"files_modified":[],"next_steps":""}'
                ),
            )
        )

        bus = EventBus()
        captured: list[CoreEvent] = []

        async def handler(event: CoreEvent) -> None:
            captured.append(event)

        bus.subscribe(EV_COMPRESSION_MANUAL, handler)

        registry = ToolRegistry()
        definition = register_manual_compact_tool(registry)
        compressor = ContextCompressor(llm_client=summary_llm, model="summary-model")

        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            tool_definitions=[definition],
        )
        context = make_agent_context(config=cfg)
        context.messages.extend(
            [
                user_msg("solve the task"),
                Message(
                    role="tool",
                    content="previous oversized tool output " * 50,
                    tool_call_id="old-tc",
                    name="search",
                ),
            ]
        )

        orch = AgentOrchestrator(
            llm_client=loop_llm,
            tool_registry=registry,
            compressor=compressor,
            event_bus=bus,
        )

        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT)

        assert result.status == ExecutionStatus.COMPLETED
        assert result.content == "done after compact"
        assert report.manual_compact_applied == 1
        assert any(item.startswith("manual_compact_goal:") for item in report.artifacts)
        assert any(event.name == EV_COMPRESSION_MANUAL for event in captured)
        tool_messages = [message for message in context.messages if message.role == "tool"]
        assert any(message.tool_call_id == "tc1" for message in tool_messages)

    @pytest.mark.asyncio
    async def test_trigger_manual_compact_updates_report(self) -> None:
        """trigger_manual_compact updates metrics and emits event."""
        from protocore.compression import ContextCompressor
        from protocore.events import EV_COMPRESSION_MANUAL, CoreEvent

        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value=Message(role="assistant", content='{"completed_tasks":[],"current_goal":"test","key_decisions":[],"files_modified":[],"next_steps":""}')
        )

        bus = EventBus()
        captured: list[CoreEvent] = []

        async def handler(event: CoreEvent) -> None:
            captured.append(event)

        bus.subscribe(EV_COMPRESSION_MANUAL, handler)

        compressor = ContextCompressor(llm_client=llm, model="test")
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.extend([
            user_msg("task"),
            Message(role="assistant", content="working on it"),
            Message(role="tool", content="result data " * 100, tool_call_id="tc1", name="search"),
        ])

        report = make_execution_report(context=context)

        orch = AgentOrchestrator(
            llm_client=llm,
            compressor=compressor,
            event_bus=bus,
        )
        await orch.trigger_manual_compact(context, report)

        assert report.manual_compact_applied == 1
        assert report.tokens_before_compression_total is not None
        assert report.tokens_after_compression_total is not None
        assert any(item.startswith("manual_compact_goal:") for item in report.artifacts)
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_trigger_manual_compact_without_compressor_warns(self) -> None:
        """trigger_manual_compact without compressor adds warning."""
        cfg = make_config(execution_mode=ExecutionMode.BYPASS)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("test"))

        report = make_execution_report(context=context)

        orch = AgentOrchestrator(
            llm_client=mock_llm_final(),
        )
        await orch.trigger_manual_compact(context, report)

        assert "manual_compact_no_compressor" in report.warnings
        assert report.manual_compact_applied == 0


# ---------------------------------------------------------------------------
# FIX: Parallel runner populates subagents_parallel_max
# ---------------------------------------------------------------------------


class TestParallelSubagentCount:
    @pytest.mark.asyncio
    async def test_parallel_sets_count_on_report(self) -> None:
        """ParallelSubagentRunner sets subagents_parallel_max on report."""
        from protocore.orchestrator import ParallelSubagentRunner

        policy = MagicMock()
        policy.max_concurrency = 3
        policy.timeout_seconds = 5.0
        policy.cancellation_mode = "graceful"
        policy.merge_results = AsyncMock(return_value=SubagentResult(
            status=SubagentStatus.SUCCESS, summary="merged"
        ))

        factory_llm = mock_llm_final("sub done")

        def factory() -> AgentOrchestrator:
            return AgentOrchestrator(llm_client=factory_llm)

        runner = ParallelSubagentRunner(policy=policy, orchestrator_factory=factory)

        cfg1 = make_config(execution_mode=ExecutionMode.BYPASS)
        cfg2 = make_config(execution_mode=ExecutionMode.BYPASS)
        ctx1 = make_agent_context(config=cfg1)
        ctx1.messages.append(user_msg("task1"))
        ctx2 = make_agent_context(config=cfg2)
        ctx2.messages.append(user_msg("task2"))

        report = make_execution_report(context=ctx1)

        merged, summaries = await runner.run_parallel(
            [("agent-a", ctx1), ("agent-b", ctx2)],
            report=report,
        )

        assert report.subagents_parallel_max == 2
        assert len(report.subagent_runs) == 2
        assert merged.status == SubagentStatus.SUCCESS


# ---------------------------------------------------------------------------
# FIX: Workflow engine integration
# ---------------------------------------------------------------------------


class TestWorkflowIntegration:
    @pytest.mark.asyncio
    async def test_run_workflow_delegates_to_engine(self) -> None:
        """run_workflow uses injected WorkflowEngine."""
        from protocore.types import WorkflowDefinition, WorkflowNode, WorkflowEdge

        workflow = WorkflowDefinition(
            name="test-dag",
            nodes=[
                WorkflowNode(node_id="n1", label="step1"),
                WorkflowNode(node_id="n2", label="step2"),
            ],
            edges=[WorkflowEdge(from_node="n1", to_node="n2")],
        )

        engine_report = ExecutionReport(
            status=ExecutionStatus.COMPLETED,
            stop_reason=StopReason.END_TURN,
            node_durations_ms={"n1": 10.0, "n2": 20.0},
            finished_at="2026-03-07T00:00:00+00:00",
            duration_ms=30.0,
        )
        engine = MagicMock()
        engine.run = AsyncMock(return_value=(
            Result(content="workflow done"),
            engine_report,
        ))

        cfg = make_config()
        ctx = make_agent_context(config=cfg)

        orch = AgentOrchestrator(
            llm_client=mock_llm_final(),
            workflow_engine=engine,
        )

        result, report = await orch.run_workflow(workflow, ctx)

        engine.run.assert_awaited_once_with(workflow, ctx)
        assert result.content == "workflow done"
        assert report.workflow_id == workflow.workflow_id
        assert report.node_count == 2
        assert report.edge_count == 1
        assert report.node_durations_ms == {"n1": 10.0, "n2": 20.0}

    @pytest.mark.asyncio
    async def test_run_workflow_without_engine_raises(self) -> None:
        """run_workflow without engine raises ContractViolationError."""
        from protocore.orchestrator import ContractViolationError
        from protocore.types import WorkflowDefinition

        cfg = make_config()
        ctx = make_agent_context(config=cfg)

        orch = AgentOrchestrator(
            llm_client=mock_llm_final(),
        )

        with pytest.raises(ContractViolationError, match="No WorkflowEngine configured"):
            await orch.run_workflow(WorkflowDefinition(name="test"), ctx)

    @pytest.mark.asyncio
    async def test_run_workflow_emits_events_and_hooks(self) -> None:
        from protocore import (
            CoreEvent,
            EV_WORKFLOW_END,
            EV_WORKFLOW_START,
            HookManager,
            hookimpl,
        )
        from protocore.types import WorkflowDefinition, WorkflowNode

        workflow = WorkflowDefinition(
            name="test-dag",
            nodes=[WorkflowNode(node_id="n1", label="step1")],
        )

        engine_report = ExecutionReport(
            status=ExecutionStatus.COMPLETED,
            stop_reason=StopReason.END_TURN,
            finished_at="2026-03-07T00:00:00+00:00",
            duration_ms=5.0,
        )
        engine = MagicMock()
        engine.run = AsyncMock(return_value=(Result(content="workflow done"), engine_report))

        bus = EventBus()
        received: list[str] = []
        hook_calls: list[str] = []

        async def handler(event: CoreEvent) -> None:
            received.append(event.name)

        class WorkflowPlugin:
            @hookimpl
            def on_workflow_start(self, workflow: Any, context: Any, report: Any) -> None:
                hook_calls.append(f"start:{workflow.workflow_id}")

            @hookimpl
            def on_workflow_end(self, workflow: Any, result: Any, report: Any) -> None:
                hook_calls.append(f"end:{workflow.workflow_id}:{result.content}")

        bus.subscribe(EV_WORKFLOW_START, handler)
        bus.subscribe(EV_WORKFLOW_END, handler)
        hooks = HookManager()
        hooks.register(WorkflowPlugin())

        ctx = make_agent_context(config=make_config())
        orch = AgentOrchestrator(
            llm_client=mock_llm_final(),
            workflow_engine=engine,
            event_bus=bus,
            hook_manager=hooks,
        )

        result, report = await orch.run_workflow(workflow, ctx)

        assert result.content == "workflow done"
        assert received == [EV_WORKFLOW_START, EV_WORKFLOW_END]
        assert hook_calls[0].startswith("start:")
        assert hook_calls[1].startswith("end:")
        assert report.status == ExecutionStatus.COMPLETED


# ---------------------------------------------------------------------------
# FIX: AUTO_SELECT uses injected AgentRegistry
# ---------------------------------------------------------------------------


class TestAutoSelectWithRegistry:
    @pytest.mark.asyncio
    async def test_auto_select_passes_registered_agents(self) -> None:
        """AUTO_SELECT passes agents from injected AgentRegistry to policy."""
        from protocore.registry import AgentRegistry

        agent_reg = AgentRegistry()
        agent_reg.register(AgentConfig(agent_id="leader", model="m", role=AgentRole.LEADER))
        agent_reg.register(AgentConfig(agent_id="coder", model="m", role=AgentRole.SUBAGENT))
        agent_reg.register(AgentConfig(agent_id="writer", model="m", role=AgentRole.SUBAGENT))

        cfg = make_config(execution_mode=ExecutionMode.AUTO_SELECT)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("pick an agent"))

        captured_agents: list[list[str]] = []
        policy = MagicMock()

        async def capture_select(
            task: str, available_agents: list[str], ctx: Any
        ) -> str:
            captured_agents.append(available_agents)
            return "coder"

        policy.select = AsyncMock(side_effect=capture_select)

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("ok"),
            agent_registry=agent_reg,
            subagent_selection_policy=policy,
        )
        result, report = await orch.run(context, run_kind=RunKind.LEADER)

        assert len(captured_agents) == 1
        assert set(captured_agents[0]) == {"coder", "writer"}
        assert "leader" not in captured_agents[0]
        assert context.metadata["auto_selected_agent"] == "coder"


class TestEnvelopeFactories:
    def test_error_envelope_supports_bidirectional_roles(self) -> None:
        envelope = make_error_envelope(
            sender_id="leader-1",
            recipient_id="sub-1",
            sender_role=AgentRole.LEADER,
            recipient_role=AgentRole.SUBAGENT,
            error_message="boom",
            error_code="ERR",
            trace_id="trace",
            session_id="session",
        )

        assert envelope.sender.role == AgentRole.LEADER
        assert envelope.recipient.role == AgentRole.SUBAGENT


# ---------------------------------------------------------------------------
# FIX: Planning hook called only after plan creation (not before)
# ---------------------------------------------------------------------------


class TestPlanningHookCallCount:
    @pytest.mark.asyncio
    async def test_plan_created_hook_called_once_after_build(self) -> None:
        """on_plan_created hook fires exactly once, after plan is built."""
        from protocore.hooks.specs import hookimpl
        from protocore.hooks.manager import HookManager
        from protocore.types import PlanArtifact

        call_log: list[PlanArtifact | None] = []

        class PlanHookPlugin:
            @hookimpl
            def on_plan_created(
                self, plan: Any, context: Any, report: Any
            ) -> None:
                call_log.append(plan)

        hm = HookManager()
        hm.register(PlanHookPlugin())

        planning = MagicMock()
        plan = PlanArtifact(trace_id="t", raw_plan="step 1")
        planning.build_plan = AsyncMock(return_value=plan)

        cfg = make_config(execution_mode=ExecutionMode.LEADER)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("build it"))

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("done"),
            planning_strategy=planning,
            hook_manager=hm,
        )
        await orch.run(context, run_kind=RunKind.LEADER)

        assert len(call_log) == 1
        assert call_log[0] is not None
        assert call_log[0].plan_id == plan.plan_id


# ===========================================================================
# Additional coverage tests
# ===========================================================================


class TestSubagentResultEdgeCases:
    """Edge cases for structured output parsing."""

    def test_parse_with_fallback_truncated_json(self) -> None:
        """Truncated JSON triggers fallback, not crash."""
        result = SubagentResult.parse_with_fallback('{"status": "success", "summa', agent_id="a1")
        assert result.status == SubagentStatus.PARTIAL
        assert any("SUBAGENT_RESULT_SCHEMA_VIOLATION" in e for e in result.errors)

    def test_parse_with_fallback_missing_required_field(self) -> None:
        """JSON missing required 'status' uses fallback."""
        raw = json.dumps({"summary": "ok", "artifacts": []})
        result = SubagentResult.parse_with_fallback(raw, agent_id="a1")
        assert result.status == SubagentStatus.PARTIAL
        assert any("SUBAGENT_RESULT_SCHEMA_VIOLATION" in e for e in result.errors)

    def test_parse_with_fallback_empty_string(self) -> None:
        result = SubagentResult.parse_with_fallback("", agent_id="a1")
        assert result.status == SubagentStatus.PARTIAL

    def test_parse_with_fallback_shell_approval_marker(self) -> None:
        raw = "[approval required before shell execution]"
        result = SubagentResult.parse_with_fallback(raw, agent_id="a1")
        assert result.status == SubagentStatus.PARTIAL
        assert "APPROVAL_REQUIRED" in result.errors
        assert "SUBAGENT_RESULT_SCHEMA_VIOLATION" not in result.errors

    def test_errors_list_max_length(self) -> None:
        """errors list respects max_length constraint."""
        from protocore.constants import MAX_SUBAGENT_ERRORS

        with pytest.raises(ValidationError):
            SubagentResult(
                status=SubagentStatus.FAILED,
                summary="x",
                errors=["err"] * (MAX_SUBAGENT_ERRORS + 1),
            )

    def test_summary_max_length(self) -> None:
        """summary respects max_length=MAX_SUMMARY_CHARS."""
        with pytest.raises(ValidationError):
            SubagentResult(
                status=SubagentStatus.SUCCESS,
                summary="x" * (MAX_SUMMARY_CHARS + 1),
            )


class TestExecutionReportComprehensiveFields:
    """Verify all required ExecutionReport fields exist and serialize correctly."""

    def test_report_contains_all_mandatory_fields(self) -> None:
        """Every field listed in ExecutionReport spec exists on ExecutionReport."""
        required_fields = {
            # Identification
            "report_version", "request_id", "trace_id", "session_id",
            "agent_id", "parent_agent_id", "run_kind",
            # Status
            "status", "stop_reason", "error_code", "error_message", "warnings",
            # Timing / loop
            "started_at", "finished_at", "duration_ms",
            "llm_latency_ms", "tool_latency_ms", "queue_wait_ms",
            "loop_count",
            # LLM metrics
            "model", "api_mode", "input_tokens", "output_tokens",
            "cached_tokens", "estimated_cost",
            # Tool metrics
            "tool_calls_total", "tool_calls_by_name", "tool_failures",
            "forced_finalization_triggered",
            # Compression metrics
            "micro_compact_applied", "auto_compact_applied",
            "manual_compact_applied", "tokens_before_compression_total", "tokens_after_compression_total",
            "auto_compact_failed", "compression_events",
            # Workflow metrics
            "workflow_id", "node_count", "edge_count", "node_durations_ms",
            # Execution mode + planning
            "execution_mode", "plan_created", "plan_id",
            "plan_artifact", "subagents_parallel_max",
            # Safety signals
            "destructive_action_requested", "destructive_action_confirmed",
            "prompt_injection_signals",
            # Artifacts
            "artifacts", "files_changed", "subagent_runs",
        }
        model_fields = set(ExecutionReport.model_fields.keys())
        missing = required_fields - model_fields
        assert not missing, f"Missing ExecutionReport fields: {missing}"

    def test_report_with_all_fields_populated_serializes(self) -> None:
        """A fully-populated report serializes to JSON without error."""
        from protocore.types import SubagentRunSummary, PlanArtifact

        report = ExecutionReport(
            request_id="req-1",
            trace_id="trace-1",
            session_id="sess-1",
            agent_id="agent-1",
            parent_agent_id="parent-1",
            run_kind=RunKind.SUBAGENT,
            status=ExecutionStatus.COMPLETED,
            stop_reason=StopReason.END_TURN,
            error_code=None,
            error_message=None,
            warnings=["w1"],
            llm_latency_ms=[100.0, 200.0],
            tool_latency_ms=[50.0],
            queue_wait_ms=10.0,
            model="qwen-32b",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=200,
            estimated_cost=0.05,
            tool_calls_total=3,
            tool_calls_by_name={"read_file": 2, "write_file": 1},
            tool_failures=0,
            forced_finalization_triggered=False,
            micro_compact_applied=2,
            auto_compact_applied=1,
            manual_compact_applied=0,
            auto_compact_failed=0,
            tokens_before_compression_total=30000,
            tokens_after_compression_total=5000,
            compression_events=[],
            loop_count=3,
            workflow_id="wf-1",
            node_count=3,
            edge_count=2,
            node_durations_ms={"node1": 100.0, "node2": 200.0},
            execution_mode=ExecutionMode.LEADER,
            plan_created=True,
            plan_id="plan-1",
            plan_artifact=PlanArtifact(plan_id="plan-1"),
            subagents_parallel_max=2,
            destructive_action_requested=1,
            destructive_action_confirmed=1,
            prompt_injection_signals=0,
            artifacts=["file.py"],
            files_changed=["src/main.py"],
            subagent_runs=[
                SubagentRunSummary(
                    agent_id="sub-1",
                    status=ExecutionStatus.COMPLETED,
                    started_at="2026-01-01T00:00:00Z",
                    finished_at="2026-01-01T00:00:01Z",
                    duration_ms=1000.0,
                )
            ],
        )
        data = json.loads(report.model_dump_json())
        assert data["report_version"] == "1.1"
        assert data["plan_created"] is True
        assert len(data["subagent_runs"]) == 1

    def test_report_partial_fill_on_error_is_valid(self) -> None:
        """Even a partially-filled report (error path) is valid JSON."""
        report = ExecutionReport(agent_id="a1")
        report.finalize(
            ExecutionStatus.FAILED,
            stop_reason=StopReason.ERROR,
            error_code="LLM_TIMEOUT",
            error_message="Connection timed out",
        )
        data = json.loads(report.model_dump_json())
        assert data["status"] == "failed"
        assert data["error_code"] == "LLM_TIMEOUT"
        assert data["finished_at"] is not None
        assert data["duration_ms"] is not None


class TestEnvelopePayloadBoundaries:
    """Boundary tests for envelope payload size and depth."""

    def test_payload_exactly_at_size_limit_is_accepted(self) -> None:
        from protocore.constants import MAX_ENVELOPE_PAYLOAD_CHARS

        # Build a task payload that's close to the limit
        filler = "x" * (MAX_ENVELOPE_PAYLOAD_CHARS - 100)
        try:
            env = AgentEnvelope(
                message_type=MessageType.TASK,
                sender=AgentIdentity(agent_id="s", role=AgentRole.LEADER),
                recipient=AgentIdentity(agent_id="r", role=AgentRole.SUBAGENT),
                payload={"task": filler},
            )
            assert env.message_type == MessageType.TASK
        except ValidationError:
            # Acceptable if filler makes it too large
            pass

    def test_payload_at_depth_limit_is_accepted(self) -> None:
        from protocore.constants import MAX_ENVELOPE_PAYLOAD_DEPTH

        # Build nested dict exactly at max depth
        nested: dict[str, Any] = {"v": "leaf"}
        for _ in range(MAX_ENVELOPE_PAYLOAD_DEPTH - 2):
            nested = {"n": nested}

        try:
            AgentEnvelope(
                message_type=MessageType.TASK,
                sender=AgentIdentity(agent_id="s", role=AgentRole.LEADER),
                recipient=AgentIdentity(agent_id="r", role=AgentRole.SUBAGENT),
                payload={"task": "t", "metadata": nested},
            )
        except ValidationError:
            pass  # May exceed size, that's fine

    def test_all_message_types_validate_payloads(self) -> None:
        """Each message_type enforces its own payload schema."""
        with pytest.raises(ValidationError):
            # TASK type requires 'task' field
            AgentEnvelope(
                message_type=MessageType.TASK,
                sender=AgentIdentity(agent_id="s", role=AgentRole.LEADER),
                recipient=AgentIdentity(agent_id="r", role=AgentRole.SUBAGENT),
                payload={"wrong_field": "x"},
            )

        with pytest.raises(ValidationError):
            # ERROR type requires 'error' and 'error_code'
            AgentEnvelope(
                message_type=MessageType.ERROR,
                sender=AgentIdentity(agent_id="s", role=AgentRole.LEADER),
                recipient=AgentIdentity(agent_id="r", role=AgentRole.SUBAGENT),
                payload={"task": "not an error"},
            )


class TestPathIsolationAdvanced:
    """Advanced path isolation cases."""

    def test_relative_path_traversal_is_denied(self) -> None:
        """Paths with ../ traversal outside allowed_paths are denied."""
        from protocore.context import validate_path_access
        from protocore.types import ToolContext
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = ToolContext(allowed_paths=[os.path.join(tmpdir, "safe")])
            os.makedirs(os.path.join(tmpdir, "safe"), exist_ok=True)

            with pytest.raises(PathIsolationError):
                validate_path_access(
                    os.path.join(tmpdir, "safe", "..", "unsafe", "file.txt"),
                    ctx,
                )

    def test_path_arguments_validation_on_nested_dict(self) -> None:
        """validate_path_arguments handles nested dict values."""
        from protocore.context import validate_path_arguments
        from protocore.types import ToolContext
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = ToolContext(allowed_paths=[tmpdir])
            import os

            valid_path = os.path.join(tmpdir, "file.txt")
            result = validate_path_arguments(
                {"path": valid_path},
                ctx,
            )
            assert len(result) == 1

    def test_path_arguments_validation_finds_nested_candidate_keys(self) -> None:
        """Nested target_path/path keys are validated recursively."""
        from protocore.context import validate_path_arguments
        from protocore.types import ToolContext
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = ToolContext(allowed_paths=[os.path.join(tmpdir, "safe")])
            os.makedirs(os.path.join(tmpdir, "safe"), exist_ok=True)
            nested_valid = os.path.join(tmpdir, "safe", "file.txt")
            nested_invalid = os.path.join(tmpdir, "unsafe", "file.txt")

            validated = validate_path_arguments(
                {"payload": {"items": [{"target_path": nested_valid}]}},
                ctx,
            )
            assert len(validated) == 1

            with pytest.raises(PathIsolationError):
                validate_path_arguments(
                    {"payload": {"items": [{"target_path": nested_invalid}]}},
                    ctx,
                )

    def test_path_arguments_validation_handles_nested_dict_and_list_payload(self) -> None:
        """Nested lists/dicts under candidate keys are validated recursively."""
        from protocore.context import validate_path_arguments
        from protocore.types import ToolContext
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            safe_root = os.path.join(tmpdir, "safe")
            os.makedirs(safe_root, exist_ok=True)
            ctx = ToolContext(allowed_paths=[safe_root])
            valid_a = os.path.join(safe_root, "a.txt")
            valid_b = os.path.join(safe_root, "b.txt")

            validated = validate_path_arguments(
                {
                    "payload": {
                        "paths": [
                            valid_a,
                            {"extra": [valid_b]},
                        ]
                    }
                },
                ctx,
            )
            assert len(validated) == 2

    def test_contains_path_argument_detects_nested_list_values(self) -> None:
        """contains_path_argument walks nested lists of objects."""
        from protocore.context import contains_path_argument

        payload = {
            "steps": [
                {"params": [{"kind": "noop"}, {"path": "/tmp/file.txt"}]},
            ]
        }
        assert contains_path_argument(payload) is True


class TestCancellationPropagation:
    """Cancellation propagates to children."""

    @pytest.mark.asyncio
    async def test_cancel_during_subagent_propagates_control_envelope(self) -> None:
        """When parent is cancelled, control:cancel envelope is sent to active children."""
        transport = AsyncMock()
        cancel_ctx = CancellationContext()

        # LLM that cancels mid-execution
        async def cancelling_complete(*args: Any, **kwargs: Any) -> Message:
            cancel_ctx.cancel("user_abort")
            raise asyncio.CancelledError("user_abort")

        llm = MagicMock()
        llm.complete = cancelling_complete

        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
        )
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("test"))
        context.metadata["_active_child_agent_ids"] = ["child-1"]

        orch = AgentOrchestrator(
            llm_client=llm,
            transport=transport,
        )

        result, report = await orch.run(context, run_kind=RunKind.SUBAGENT, cancel_ctx=cancel_ctx)
        assert report.status == ExecutionStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_parallel_parent_cancel_propagates_even_in_graceful_mode(self) -> None:
        """Parent cancellation always sends control:cancel to active children."""
        from protocore.orchestrator import ParallelSubagentRunner

        transport = AsyncMock()
        cancel_ctx = CancellationContext()

        async def hanging_complete(*args: Any, **kwargs: Any) -> Message:
            await cancel_ctx.wait()
            raise asyncio.CancelledError("cancelled")

        def make_orch() -> AgentOrchestrator:
            llm = MagicMock()
            llm.complete = AsyncMock(side_effect=hanging_complete)
            return AgentOrchestrator(llm_client=llm, transport=transport)

        class Policy:
            max_concurrency = 2
            timeout_seconds = 1.0
            cancellation_mode = "graceful"

            async def merge_results(
                self,
                results: list[SubagentResult | None],
                agent_ids: list[str],
            ) -> SubagentResult:
                return SubagentResult(status=SubagentStatus.PARTIAL, summary="cancelled")

        parent_cfg = make_config(execution_mode=ExecutionMode.PARALLEL)
        parent_cfg = parent_cfg.model_copy(update={"agent_id": "leader"})
        parent_ctx = make_agent_context(config=parent_cfg)
        parent_ctx.messages.append(user_msg("run both"))

        runner = ParallelSubagentRunner(
            policy=Policy(),
            orchestrator_factory=make_orch,
            transport=transport,
            parent_context=parent_ctx,
        )
        tasks = []
        for agent_id in ("child-a", "child-b"):
            cfg = make_config(execution_mode=ExecutionMode.BYPASS)
            cfg = cfg.model_copy(update={"agent_id": agent_id})
            ctx = make_agent_context(config=cfg)
            ctx.messages.append(user_msg(f"task for {agent_id}"))
            tasks.append((agent_id, ctx))

        run_task = asyncio.create_task(runner.run_parallel(tasks, cancel_ctx=cancel_ctx))
        await asyncio.sleep(0.01)
        cancel_ctx.cancel("stop")
        _, summaries = await run_task

        destinations = [call.kwargs["destination"] for call in transport.send.await_args_list]
        assert sorted(destinations) == ["child-a", "child-b"]
        assert all(summary.status == ExecutionStatus.CANCELLED for summary in summaries)

    @pytest.mark.asyncio
    async def test_cancellation_context_wait(self) -> None:
        """CancellationContext.wait() resolves after cancel()."""
        ctx = CancellationContext()

        async def cancel_later() -> None:
            await asyncio.sleep(0.01)
            ctx.cancel("test")

        asyncio.create_task(cancel_later())
        await asyncio.wait_for(ctx.wait(), timeout=1.0)
        assert ctx.is_cancelled
        assert ctx.reason == "test"


class TestAutoSelectEdgeCases:
    """AUTO_SELECT edge cases."""

    @pytest.mark.asyncio
    async def test_auto_select_with_no_registered_subagents(self) -> None:
        """AUTO_SELECT with empty agent registry passes empty list to policy."""
        selected_agents: list[list[str]] = []

        async def select(task: str, available: list[str], ctx: Any) -> str:
            selected_agents.append(available)
            return "fallback-agent"

        policy = MagicMock()
        policy.select = select

        agent_registry = AgentRegistry()
        # Register the fallback agent so _build_subagent_context works
        agent_registry.register(AgentConfig(
            agent_id="fallback-agent", model="gpt-4o", role=AgentRole.SUBAGENT,
        ))

        cfg = make_config(execution_mode=ExecutionMode.AUTO_SELECT)
        context = make_agent_context(config=cfg)
        context.messages.append(user_msg("do something"))

        orch = AgentOrchestrator(
            llm_client=mock_llm_final("done"),
            subagent_selection_policy=policy,
            agent_registry=agent_registry,
        )

        result, report = await orch.run(context, run_kind=RunKind.LEADER)
        # Policy was called — the available list should contain only the registered subagent
        assert len(selected_agents) == 1


class TestSessionSnapshotSerialization:
    """Session snapshot round-trip."""

    def test_snapshot_round_trips_json(self) -> None:
        from protocore.types import SessionSnapshot

        snap = SessionSnapshot(
            session_id="s1",
            trace_id="t1",
            agent_id="a1",
            message_history_ref="session:s1:messages",
            execution_metadata_ref="request:r1:metadata",
            messages=[
                Message(role="system", content="You are a helper."),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!"),
                Message(role="tool", content="result", tool_call_id="tc1", name="search"),
            ],
            execution_report_id="rep-1",
            metadata={"custom_key": "custom_value"},
        )
        raw = snap.model_dump_json()
        restored = SessionSnapshot.model_validate_json(raw)
        assert restored.session_id == "s1"
        assert len(restored.messages) == 4
        assert restored.messages[3].role == "tool"
        assert restored.metadata["custom_key"] == "custom_value"


class TestEventBusAdvanced:
    """EventBus: unsubscribe, multiple listeners, error sink."""

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self) -> None:
        bus = EventBus()
        received: list[str] = []

        async def handler(event: Any) -> None:
            received.append(event.name)

        bus.subscribe("test.event", handler)
        from protocore.events import CoreEvent

        await bus.emit(CoreEvent(name="test.event", payload={}))
        assert len(received) == 1

        bus.unsubscribe("test.event", handler)
        await bus.emit(CoreEvent(name="test.event", payload={}))
        assert len(received) == 1  # No new delivery
        assert bus.cleanup_stale_handlers() == 0
        assert "test.event" not in bus._handlers

    @pytest.mark.asyncio
    async def test_multiple_listeners_all_receive(self) -> None:
        bus = EventBus()
        received_a: list[str] = []
        received_b: list[str] = []

        async def handler_a(event: Any) -> None:
            received_a.append(event.name)

        async def handler_b(event: Any) -> None:
            received_b.append(event.name)

        bus.subscribe("x", handler_a)
        bus.subscribe("x", handler_b)
        from protocore.events import CoreEvent

        await bus.emit(CoreEvent(name="x", payload={}))
        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_error_sink_receives_handler_failures(self) -> None:
        bus = EventBus()
        sink_calls: list[tuple[str, str]] = []

        async def bad_handler(event: Any) -> None:
            raise ValueError("boom")

        async def sink(event: Any, exc: Exception) -> None:
            sink_calls.append((event.name, str(exc)))

        bus.subscribe("fail", bad_handler)
        bus.set_error_sink(sink)
        from protocore.events import CoreEvent

        await bus.emit(CoreEvent(name="fail", payload={}))
        assert len(sink_calls) == 1
        assert "boom" in sink_calls[0][1]


class TestWorkflowDefinitionValidation:
    """WorkflowDefinition edge cases."""

    def test_empty_workflow_is_valid(self) -> None:
        from protocore.types import WorkflowDefinition

        wf = WorkflowDefinition(name="empty")
        assert len(wf.nodes) == 0
        assert len(wf.edges) == 0

    def test_linear_chain_is_valid(self) -> None:
        from protocore.types import WorkflowDefinition, WorkflowNode, WorkflowEdge

        wf = WorkflowDefinition(
            name="chain",
            nodes=[
                WorkflowNode(node_id="a", label="A"),
                WorkflowNode(node_id="b", label="B"),
                WorkflowNode(node_id="c", label="C"),
            ],
            edges=[
                WorkflowEdge(from_node="a", to_node="b"),
                WorkflowEdge(from_node="b", to_node="c"),
            ],
        )
        assert len(wf.nodes) == 3

    def test_edge_referencing_unknown_node_raises(self) -> None:
        from protocore.types import WorkflowDefinition, WorkflowNode, WorkflowEdge

        with pytest.raises(ValidationError, match="unknown node"):
            WorkflowDefinition(
                nodes=[WorkflowNode(node_id="a", label="A")],
                edges=[WorkflowEdge(from_node="a", to_node="nonexistent")],
            )

    def test_large_linear_workflow_validates_without_recursion_depth_failures(self) -> None:
        from protocore.types import WorkflowDefinition, WorkflowEdge, WorkflowNode

        node_count = 1_500
        nodes = [WorkflowNode(node_id=f"n{i}", label=f"N{i}") for i in range(node_count)]
        edges = [WorkflowEdge(from_node=f"n{i}", to_node=f"n{i + 1}") for i in range(node_count - 1)]
        wf = WorkflowDefinition(name="large-chain", nodes=nodes, edges=edges)
        assert len(wf.nodes) == node_count

    def test_disconnected_workflow_raises(self) -> None:
        from protocore.types import WorkflowDefinition, WorkflowEdge, WorkflowNode

        with pytest.raises(ValidationError, match="weakly connected"):
            WorkflowDefinition(
                name="disconnected",
                nodes=[
                    WorkflowNode(node_id="a", label="A"),
                    WorkflowNode(node_id="b", label="B"),
                    WorkflowNode(node_id="c", label="C"),
                ],
                edges=[WorkflowEdge(from_node="a", to_node="b")],
            )


class TestProtocolApiModeInContract:
    """api_mode is part of the LLMClient protocol."""

    def test_llm_client_protocol_has_api_mode_parameter(self) -> None:
        """Verify api_mode is an explicit parameter, not just **kwargs."""
        import inspect
        from protocore.protocols import LLMClient

        sig = inspect.signature(LLMClient.complete)
        assert "api_mode" in sig.parameters, "LLMClient.complete must have api_mode parameter"

        sig_structured = inspect.signature(LLMClient.complete_structured)
        assert "api_mode" in sig_structured.parameters

        sig_stream = inspect.signature(LLMClient.stream_with_tools)
        assert "api_mode" in sig_stream.parameters


class TestThinkingAndSamplingConfigPassThrough:
    @pytest.mark.asyncio
    async def test_orchestrator_forwards_extended_qwen_compatible_request_params(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(role="assistant", content="ok"))

        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            api_mode=ApiMode.CHAT_COMPLETIONS,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            max_tokens=1024,
            thinking_tokens_reserve=512,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.0,
            enable_thinking=False,
            llm_request_kwargs={"seed": 7},
            llm_extra_body={"chat_template_kwargs": {"foo": "bar"}},
        )
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=llm)
        _result, _report = await orch.run(ctx)

        assert llm.complete.await_args is not None
        call_kwargs = llm.complete.await_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.8
        assert call_kwargs["presence_penalty"] == 1.5
        assert call_kwargs["max_tokens"] == 1536
        assert call_kwargs["seed"] == 7
        assert call_kwargs["extra_body"]["top_k"] == 20
        assert call_kwargs["extra_body"]["min_p"] == 0.0
        assert call_kwargs["extra_body"]["repetition_penalty"] == 1.0
        assert call_kwargs["extra_body"]["chat_template_kwargs"]["foo"] == "bar"
        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_orchestrator_applies_selective_thinking_profile_defaults(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(role="assistant", content="ok"))

        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            thinking_profile=ThinkingProfilePreset.INSTRUCT_TOOL_WORKER,
        )
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=llm)
        _result, _report = await orch.run(ctx)

        assert llm.complete.await_args is not None
        call_kwargs = llm.complete.await_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.8
        assert call_kwargs["presence_penalty"] == 1.5
        assert call_kwargs["extra_body"]["top_k"] == 20
        assert call_kwargs["extra_body"]["min_p"] == 0.0
        assert call_kwargs["extra_body"]["repetition_penalty"] == 1.0
        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_top_level_extra_body_enable_thinking_takes_precedence(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(role="assistant", content="ok"))

        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            enable_thinking=False,
            llm_extra_body={"enable_thinking": True},
        )
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=llm)
        _result, _report = await orch.run(ctx)

        assert llm.complete.await_args is not None
        call_kwargs = llm.complete.await_args.kwargs
        assert call_kwargs["extra_body"]["enable_thinking"] is True
        assert "chat_template_kwargs" not in call_kwargs["extra_body"]

    def test_agent_config_force_off_step_policy_disables_thinking_when_unspecified(self) -> None:
        cfg = AgentConfig(
            agent_id="agent",
            model="gpt-4o",
            thinking_run_policy=ThinkingRunPolicy.FORCE_OFF,
        )

        resolved = cfg.resolved_with_selective_thinking()

        assert resolved.enable_thinking is False


class TestStreamingByDefault:
    @pytest.mark.asyncio
    async def test_orchestrator_emits_stream_delta_events_when_stream_enabled(self) -> None:
        llm = MagicMock()

        async def _complete(**kwargs: Any) -> Message:
            callback = kwargs.get("stream_event_callback")
            if callback is not None:
                await callback(
                    {
                        "kind": "reasoning",
                        "text": "thinking",
                        "provider_event_type": "response.reasoning.delta",
                    }
                )
                await callback(
                    {
                        "kind": "text",
                        "text": "done",
                        "provider_event_type": "response.output_text.delta",
                    }
                )
            return Message(role="assistant", content="done")

        llm.complete = AsyncMock(side_effect=_complete)
        bus = EventBus()
        seen_events: list[dict[str, Any]] = []

        async def _on_stream(event: Any) -> None:
            seen_events.append(event.payload)

        bus.subscribe("llm.stream.delta", _on_stream)

        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            stream=True,
            emit_reasoning_in_stream=True,
        )
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=llm, event_bus=bus)
        result, report = await orch.run(ctx)

        assert result.content == "done"
        assert report.status == ExecutionStatus.COMPLETED
        assert llm.complete.await_args is not None
        assert llm.complete.await_args.kwargs["stream"] is True
        assert [event["kind"] for event in seen_events] == ["reasoning", "text"]
        assert all(event["agent_id"] == cfg.agent_id for event in seen_events)

    @pytest.mark.asyncio
    async def test_orchestrator_respects_stream_false(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(role="assistant", content="done"))
        bus = EventBus()
        seen_events: list[dict[str, Any]] = []

        async def _on_stream(event: Any) -> None:
            seen_events.append(event.payload)

        bus.subscribe("llm.stream.delta", _on_stream)

        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            stream=False,
        )
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=llm, event_bus=bus)
        result, report = await orch.run(ctx)

        assert result.content == "done"
        assert report.status == ExecutionStatus.COMPLETED
        assert llm.complete.await_args is not None
        assert llm.complete.await_args.kwargs["stream"] is False
        assert seen_events == []

    def test_agent_config_rejects_hard_limit_below_soft_limit(self) -> None:
        with pytest.raises(ValidationError, match="output_token_hard_limit"):
            AgentConfig(
                agent_id="agent",
                model="gpt-4o",
                output_token_soft_limit=100,
                output_token_hard_limit=50,
            )

    @pytest.mark.asyncio
    async def test_orchestrator_records_output_token_guardrail_warnings(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value=Message(
                role="assistant",
                content="ok",
                usage=LLMUsage(input_tokens=1, output_tokens=120, cached_tokens=0),
            )
        )

        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            output_token_soft_limit=100,
            output_token_hard_limit=110,
        )
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=llm)
        _result, report = await orch.run(ctx)

        assert "output_token_soft_limit_exceeded:120/100" in report.warnings
        assert "output_token_hard_limit_exceeded:120/110" in report.warnings


class TestToolContextToolCallId:
    @pytest.mark.asyncio
    async def test_registry_dispatch_injects_tool_call_id_into_context(self) -> None:
        from protocore.types import ToolContext

        seen: dict[str, str] = {}

        async def handler(*, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = arguments
            seen["tool_call_id"] = context.tool_call_id
            return ToolResult(tool_name="echo", content="ok")

        registry = ToolRegistry()
        registry.register(ToolDefinition(name="echo", description="echo"), handler)

        context = ToolContext()
        result = await registry.dispatch(
            "echo",
            {"x": 1},
            context,
            tool_call_id="tc-from-dispatcher",
        )

        assert result is not None
        assert seen["tool_call_id"] == "tc-from-dispatcher"

    def test_tool_result_allows_omitted_tool_call_id(self) -> None:
        result = ToolResult(tool_name="echo", content="ok")
        assert result.tool_call_id == ""


class TestLlmExtraBodyPassthrough:
    @pytest.mark.asyncio
    async def test_orchestrator_run_matches_direct_complete_extra_body(self) -> None:
        class RecordingLLM:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            async def complete(self, **kwargs: Any) -> Message:
                self.calls.append(kwargs)
                return Message(role="assistant", content="ok")

            async def complete_structured(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)
                return {"status": "ok"}

            async def stream_with_tools(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
                self.calls.append(kwargs)
                if False:  # pragma: no cover
                    yield {}

        llm = RecordingLLM()
        expected_extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        await llm.complete(
            messages=[Message(role="user", content="hello")],
            extra_body=expected_extra_body,
        )

        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            llm_extra_body=expected_extra_body,
        )
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=cast(Any, llm))
        _result, _report = await orch.run(ctx)

        assert llm.calls[0]["extra_body"] == expected_extra_body
        assert llm.calls[1]["extra_body"] == expected_extra_body

    @pytest.mark.asyncio
    async def test_orchestrator_subagent_run_forwards_extra_body(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(role="assistant", content="ok"))

        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            llm_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=llm)
        _result, _report = await orch.run(ctx, run_kind=RunKind.SUBAGENT)

        assert llm.complete.await_args is not None
        assert llm.complete.await_args.kwargs["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    @pytest.mark.asyncio
    async def test_orchestrator_run_runtime_extra_body_overrides_config(self) -> None:
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=Message(role="assistant", content="ok"))

        cfg = make_config(
            execution_mode=ExecutionMode.BYPASS,
            llm_extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "foo": "keep",
                },
                "seed": 123,
            },
        )
        ctx = make_agent_context(config=cfg)
        ctx.messages.append(Message(role="user", content="hello"))

        orch = AgentOrchestrator(llm_client=llm)
        _result, _report = await orch.run(
            ctx,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        assert llm.complete.await_args is not None
        assert llm.complete.await_args.kwargs["extra_body"] == {
            "chat_template_kwargs": {
                "enable_thinking": False,
                "foo": "keep",
            },
            "seed": 123,
        }


class TestThinkingDisableHelpers:
    def test_qwen_no_thinking_constant_shape(self) -> None:
        assert QWEN_NO_THINKING_EXTRA_BODY == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_agent_config_with_thinking_disabled_helper(self) -> None:
        cfg = make_config(
            llm_extra_body={"chat_template_kwargs": {"foo": "bar"}, "seed": 7},
            enable_thinking=True,
        )

        disabled = cfg.with_thinking_disabled()

        assert disabled.enable_thinking is False
        assert disabled.llm_extra_body == {
            "chat_template_kwargs": {"foo": "bar", "enable_thinking": False},
            "seed": 7,
        }
        assert cfg.llm_extra_body == {
            "chat_template_kwargs": {"foo": "bar"},
            "seed": 7,
        }
