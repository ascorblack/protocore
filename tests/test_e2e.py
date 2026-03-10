from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from protocore import (
    AgentConfig,
    AgentOrchestrator,
    ExecutionMode,
    ExecutionStatus,
    Message,
    StopReason,
    make_agent_context,
)
from protocore.hooks.manager import HookManager
from protocore.hooks.specs import hookimpl
from protocore.registry import AgentRegistry


@pytest.mark.asyncio
async def test_leader_subagent_full_chain_e2e() -> None:
    leader_cfg = AgentConfig(
        agent_id="leader-1",
        model="test-model",
        execution_mode=ExecutionMode.AUTO_SELECT,
    )
    sub_cfg = AgentConfig(
        agent_id="sub-1",
        model="test-model",
        execution_mode=ExecutionMode.BYPASS,
    )
    context = make_agent_context(config=leader_cfg)
    context.messages.append(Message(role="user", content="Implement a small fix and summarize it."))

    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=Message(
            role="assistant",
            content=json.dumps(
                {
                    "status": "success",
                    "summary": "Subagent completed task successfully.",
                    "artifacts": ["diff:fix.patch"],
                    "files_changed": ["src/example.py"],
                    "tool_calls_made": 1,
                    "errors": [],
                    "next_steps": "Run tests.",
                }
            ),
        )
    )

    selected: list[tuple[str, list[str], str]] = []
    selection_policy = MagicMock()

    async def select_agent(task: str, available_agents: list[str], _ctx: Any) -> str:
        selected.append((task, available_agents, "sub-1"))
        return "sub-1"

    selection_policy.select = AsyncMock(side_effect=select_agent)

    seen_envelopes: list[dict[str, Any]] = []

    class CapturePlugin:
        @hookimpl
        def on_subagent_start(
            self, agent_id: str, envelope_payload: dict[str, Any], report: Any
        ) -> None:
            seen_envelopes.append({"agent_id": agent_id, "payload": envelope_payload})

    hooks = HookManager()
    hooks.register(CapturePlugin())

    registry = AgentRegistry()
    registry.register(sub_cfg)

    orch = AgentOrchestrator(
        llm_client=llm,
        agent_registry=registry,
        subagent_selection_policy=selection_policy,
        hook_manager=hooks,
    )
    result, report = await orch.run(context)

    assert len(selected) == 1
    assert selected[0][1] == ["sub-1"]
    assert len(seen_envelopes) == 1
    assert seen_envelopes[0]["agent_id"] == "sub-1"
    assert seen_envelopes[0]["payload"]["task"] == "Implement a small fix and summarize it."

    assert result.content == "Subagent completed task successfully."
    assert result.status == ExecutionStatus.COMPLETED
    assert report.status == ExecutionStatus.COMPLETED
    assert report.stop_reason == StopReason.END_TURN
    assert len(report.subagent_runs) == 1
    assert any(item == "subagent:sub-1" for item in report.artifacts)
    assert any(item == "auto_selected_agent:sub-1" for item in report.artifacts)
    assert any(item == "diff:fix.patch" for item in report.artifacts)
    assert report.files_changed == ["src/example.py"]
