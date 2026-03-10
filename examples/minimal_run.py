"""Минимальный цикл оркестратора с мок-LLM (без внешнего API).

Запуск: uv run python examples/minimal_run.py
"""
from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from protocore import (
    AgentConfig,
    AgentOrchestrator,
    Message,
    make_agent_context,
)
from protocore.protocols import LLMClient
from protocore.types import ExecutionMode


def make_mock_llm(response_text: str = "Привет! Я мок-модель. Готов к работе.") -> LLMClient:
    """Мок LLM: один вызов complete возвращает финальный ответ без tool calls."""
    llm = MagicMock(spec=LLMClient)
    llm.complete = AsyncMock(
        return_value=Message(role="assistant", content=response_text)
    )
    return cast(LLMClient, llm)


async def main() -> None:
    llm = make_mock_llm()
    config = AgentConfig(
        name="minimal-demo",
        model="mock",
        execution_mode=ExecutionMode.BYPASS,
        system_prompt="Ты полезный ассистент.",
    )
    context = make_agent_context(config=config)
    context.messages.append(Message(role="user", content="Скажи коротко привет."))

    orchestrator = AgentOrchestrator(llm_client=llm)
    result, report = await orchestrator.run(context)

    print("Ответ:", result.content)
    print(f"Токены: {report.input_tokens} вх, {report.output_tokens} вых")
    print(f"Статус: {report.status}, stop_reason: {report.stop_reason}")


if __name__ == "__main__":
    asyncio.run(main())
