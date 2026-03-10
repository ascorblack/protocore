from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from protocore import ApiMode, Message, OpenAILLMClient
from protocore.constants import ThinkingProfileRegistry
from protocore.context import build_tool_context, validate_path_access
from protocore.orchestrator_utils import recover_tool_calls_from_assistant_text
from protocore.shell_safety import DefaultShellSafetyPolicy
from protocore.types import (
    AgentConfig,
    AgentContext,
    AgentContextMeta,
    PolicyDecision,
    ShellExecutionRequest,
    ShellToolConfig,
    ShellToolProfile,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolContext,
)


def _tool_context(allowed_paths: list[str] | None = None) -> ToolContext:
    return ToolContext(
        session_id="session-1",
        trace_id="trace-1",
        agent_id="agent-1",
        allowed_paths=allowed_paths or [],
        metadata={},
    )


def test_recover_tool_calls_limits_recovered_candidates() -> None:
    tools = [ToolDefinition(name="search", description="search")]
    text = "\n".join(
        [
            f'```json\n{{"name":"search","arguments":{{"q":"item-{index}"}}}}\n```'
            for index in range(8)
        ]
    )

    recovered = recover_tool_calls_from_assistant_text(
        text,
        tools,
        max_candidates=3,
    )

    assert len(recovered) == 3
    assert [call["function"]["name"] for call in recovered] == ["search"] * 3


def test_agent_config_disables_fallback_tool_recovery_by_default() -> None:
    cfg = AgentConfig(agent_id="agent", model="gpt-4o")
    assert cfg.allow_fallback_tool_call_recovery is False


def test_validate_path_access_logs_allowed_path_fallback(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    ctx = _tool_context([str(tmp_path)])

    with caplog.at_level("DEBUG"):
        resolved = validate_path_access("nested", ctx)

    assert resolved == nested.resolve()
    assert "using first allowed_path as root" in caplog.text


def test_build_tool_context_resolves_paths_and_warns_for_missing_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    with caplog.at_level("WARNING"):
        ctx = build_tool_context(
            session_id="session",
            trace_id="trace",
            agent_id="agent",
            allowed_paths=["relative-dir", "missing-dir"],
        )

    assert all(Path(path).is_absolute() for path in ctx.allowed_paths)
    assert Path(ctx.allowed_paths[0]).name == "relative-dir"
    assert "allowed_path does not exist" in caplog.text


def test_make_agent_context_rejects_unsafe_external_ids() -> None:
    from protocore.factories import make_agent_context

    cfg = AgentConfig(agent_id="agent", model="gpt-4o")

    with pytest.raises(ValueError, match="session_id must be a valid UUID or a safe opaque identifier"):
        make_agent_context(config=cfg, session_id="../escape")


def test_provider_call_id_aliases_normalize_to_internal_fields() -> None:
    tool_call = ToolCall.model_validate(
        {
            "call_id": "call-1",
            "function": {"name": "search", "arguments": "{}"},
        }
    )
    message = Message(role="tool", content="done", tool_call_id="call-1", name="search")
    result = ToolResult.model_validate(
        {"call_id": "call-1", "tool_name": "search", "content": "done"}
    )

    assert tool_call.id == "call-1"
    assert tool_call.call_id == "call-1"
    assert message.tool_call_id == "call-1"
    assert message.call_id == "call-1"
    assert result.tool_call_id == "call-1"
    assert result.call_id == "call-1"


def test_agent_context_migrates_legacy_active_child_agent_ids() -> None:
    ctx = AgentContext(
        session_id="11111111-1111-1111-1111-111111111111",
        trace_id="22222222-2222-2222-2222-222222222222",
        request_id="33333333-3333-3333-3333-333333333333",
        config=AgentConfig(agent_id="agent", model="gpt-4o"),
        tool_context=ToolContext(),
        metadata={"_active_child_agent_ids": ["child-a"]},
    )

    assert ctx.metadata[AgentContextMeta.ACTIVE_CHILD_AGENT_IDS.value] == ["child-a"]
    assert ctx.metadata[AgentContextMeta.LEGACY_ACTIVE_CHILD_AGENT_IDS.value] == ["child-a"]


def test_thinking_profiles_planner_and_analytical_are_distinct() -> None:
    planner = ThinkingProfileRegistry.get("thinking_planner")
    analytical = ThinkingProfileRegistry.get("thinking_analytical")

    assert planner is not None
    assert analytical is not None
    assert planner != analytical
    assert planner["temperature"] > analytical["temperature"]


@pytest.mark.asyncio
async def test_default_shell_policy_explain_decision_reports_reason() -> None:
    policy = DefaultShellSafetyPolicy()
    request = ShellExecutionRequest(command="touch file.txt")
    capability = ShellToolConfig(profile=ShellToolProfile.READ_ONLY)

    decision = await policy.evaluate(request, _tool_context(["/tmp"]), capability)
    explanation = policy.explain_decision(request, _tool_context(["/tmp"]), capability)

    assert decision == PolicyDecision.DENY
    assert explanation["decision"] == PolicyDecision.DENY.value
    reasons = explanation.get("reasons", [])
    assert isinstance(reasons, list)
    assert "mutating_command_in_read_only_profile" in reasons


@pytest.mark.asyncio
async def test_openai_client_strips_logging_context_before_sdk_call() -> None:
    client = OpenAILLMClient(api_key="test", api_mode=ApiMode.RESPONSES)
    response = MagicMock()
    response.output = []
    response.output_text = "ok"

    with patch.object(
        client._client.responses,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = response
        result = await client.complete(
            messages=[Message(role="user", content="hi")],
            model="gpt-4o",
            stream=False,
            logging_context={"request_id": "req-1", "session_id": "sess-1"},
        )

    assert result.content == "ok"
    assert mock_create.await_args is not None
    assert "logging_context" not in mock_create.await_args.kwargs
