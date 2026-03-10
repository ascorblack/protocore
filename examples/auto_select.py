"""Пример AUTO_SELECT с выбором сабагента по описанию (мок-LLM, без API).

Запуск: uv run python examples/auto_select.py
"""
from __future__ import annotations

import asyncio

from protocore import (
    AgentConfig,
    AgentOrchestrator,
    AgentRegistry,
    ExecutionMode,
    Message,
    SubagentResult,
    SubagentStatus,
    make_agent_context,
)
from protocore.testing import FakeLLMClient


def make_fake_llm() -> FakeLLMClient:
    """Очередь ответов: выбор сабагента -> ответ сабагента."""
    selection = Message(role="assistant", content='{"agent_id":"coder"}')
    subagent_payload = SubagentResult(
        status=SubagentStatus.SUCCESS,
        summary="Implemented the requested change.",
        artifacts=[],
        files_changed=[],
        tool_calls_made=0,
        errors=[],
        next_steps=None,
    ).model_dump_json()
    subagent_reply = Message(role="assistant", content=subagent_payload)
    return FakeLLMClient(complete_responses=[selection, subagent_reply])


async def main() -> None:
    llm = make_fake_llm()

    registry = AgentRegistry()
    registry.register(
        AgentConfig(
            agent_id="coder",
            name="Coder",
            description="Code generation and debugging",
            model="mock",
        )
    )
    registry.register(
        AgentConfig(
            agent_id="writer",
            name="Writer",
            description="Release notes and documentation",
            model="mock",
        )
    )

    leader_cfg = AgentConfig(
        name="leader",
        model="mock",
        execution_mode=ExecutionMode.AUTO_SELECT,
        system_prompt="You are a delegating leader.",
    )
    context = make_agent_context(config=leader_cfg)
    context.messages.append(Message(role="user", content="Implement the fix"))

    orchestrator = AgentOrchestrator(llm_client=llm, agent_registry=registry)
    result, report = await orchestrator.run(context)

    print("Ответ:", result.content)
    print("Статус:", report.status)
    print("Auto-selected:", context.metadata.get("auto_selected_agent"))


if __name__ == "__main__":
    asyncio.run(main())
