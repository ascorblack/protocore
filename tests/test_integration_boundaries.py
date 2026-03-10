from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from protocore.integrations.openai import OpenAILLMClient as CanonicalOpenAIClient
from protocore import (
    AgentConfig,
    AgentOrchestrator,
    ExecutionMode,
    Message,
    RunKind,
    ShellAccessMode,
    ShellToolConfig,
    ToolContextMeta,
    ToolDefinition,
    make_agent_context,
    make_execution_report,
)


def test_canonical_openai_integration_import_matches_legacy_export() -> None:
    from protocore import OpenAILLMClient as LegacyOpenAIClient

    assert CanonicalOpenAIClient is LegacyOpenAIClient


def test_agent_context_mirrors_session_refs_into_tool_metadata() -> None:
    ctx = make_agent_context(
        config=AgentConfig(agent_id="agent", model="test-model"),
        session_id="session-1",
        request_id="request-1",
    )

    assert ctx.tool_context.metadata["message_history_ref"] == "session:session-1:messages"
    assert ctx.tool_context.metadata["execution_metadata_ref"] == "request:request-1:metadata"
    assert ctx.tool_context.metadata["request_id"] == "request-1"


@pytest.mark.asyncio
async def test_hidden_tool_is_denied_by_runtime_visibility_allowlist() -> None:
    llm = MagicMock()
    orch = AgentOrchestrator(llm_client=llm)
    ctx = make_agent_context(
        config=AgentConfig(
            agent_id="agent",
            model="test-model",
            tool_definitions=[ToolDefinition(name="visible", description="visible")],
        )
    )
    ctx.tool_context.metadata[ToolContextMeta.VISIBLE_TOOL_NAMES] = ["visible"]
    report = make_execution_report(context=ctx, run_kind=RunKind.LEADER)

    result = await orch._dispatch_tool(
        {
            "id": "tc-1",
            "function": {"name": "hidden", "arguments": "{}"},
        },
        ctx,
        report,
        RunKind.LEADER,
    )

    assert result.is_error is True
    assert "DENIED by tool_visibility preflight" in result.content


@pytest.mark.asyncio
async def test_runtime_tool_cap_records_warning_and_counts_shell() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=Message(role="assistant", content="done"))
    shell_executor = MagicMock()
    cfg = AgentConfig(
        agent_id="agent",
        model="test-model",
        execution_mode=ExecutionMode.BYPASS,
        max_visible_tools=2,
        shell_tool_config=ShellToolConfig(access_mode=ShellAccessMode.ALL_AGENTS),
        tool_definitions=[
            ToolDefinition(name="tool-a", description="a"),
            ToolDefinition(name="tool-b", description="b"),
        ],
    )
    ctx = make_agent_context(config=cfg)
    ctx.messages.append(Message(role="user", content="hello"))

    orch = AgentOrchestrator(llm_client=llm, shell_executor=shell_executor)
    _, report = await orch.run(ctx)

    assert "runtime_tools_capped:2/3" in report.warnings
    assert ctx.tool_context.metadata[ToolContextMeta.VISIBLE_TOOL_NAMES] == [
        "tool-a",
        "shell_exec",
    ]
