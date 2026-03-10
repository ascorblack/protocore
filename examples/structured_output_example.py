"""Пример структурированного вывода по pydantic-схеме (мок-LLM, без API).

Мок возвращает валидный JSON, соответствующий схеме.
Запуск: uv run python examples/structured_output_example.py
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from protocore import Message, make_agent_context
from protocore.integrations.llm.openai_client import OpenAILLMClient
from protocore.protocols import LLMClient
from protocore.types import AgentConfig


class TaskBreakdown(BaseModel):
    """Пример схемы: разбиение задачи на шаги."""

    title: str
    steps: list[str]
    estimated_minutes: int


class Choice(BaseModel):
    """Пример схемы: выбор варианта с оценкой."""

    option: str
    score: float


def make_mock_llm_structured() -> LLMClient:
    """Мок LLM: complete_structured возвращает экземпляр Choice."""
    llm = MagicMock(spec=OpenAILLMClient)

    async def complete_structured(*args: Any, **kwargs: Any) -> Choice:
        return Choice(option="A", score=0.9)

    async def complete(*args: Any, **kwargs: Any) -> Message:
        return Message(role="assistant", content="ok")

    llm.complete_structured = AsyncMock(side_effect=complete_structured)
    llm.complete = AsyncMock(side_effect=complete)
    return cast(LLMClient, llm)


async def main() -> None:
    # Вариант 1: прямой вызов complete_structured с моком
    mock = make_mock_llm_structured()
    messages = [Message(role="user", content="Выбери A или B")]
    result = await mock.complete_structured(
        messages=messages,
        schema=Choice,
        system="You are a classifier.",
    )
    print("Структурированный результат (Choice):", result)
    assert result.option == "A"
    assert result.score == 0.9

    # Вариант 2: парсинг сырого JSON в pydantic-модель (как делает OpenAILLMClient)
    config = AgentConfig(name="demo", model="mock")
    context = make_agent_context(config=config)
    context.messages.append(
        Message(role="user", content="Разбей задачу «написать отчёт» на шаги")
    )
    raw_json = json.dumps({
        "title": "Написать отчёт",
        "steps": ["Сбор данных", "Черновик", "Ревью", "Финальная версия"],
        "estimated_minutes": 60,
    })
    parsed = TaskBreakdown.model_validate_json(raw_json)
    print("Структурированный результат (TaskBreakdown):", parsed)
    print("  шаги:", parsed.steps)
    print("  минут:", parsed.estimated_minutes)


if __name__ == "__main__":
    asyncio.run(main())
