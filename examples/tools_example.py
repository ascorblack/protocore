"""Пример регистрации инструментов и одного запроса с tool call (мок-LLM, без API).

Мок возвращает сначала вызов инструмента echo, затем финальный ответ после tool result.
Запуск: uv run python examples/tools_example.py
"""
from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from protocore import (
    AgentConfig,
    AgentOrchestrator,
    Message,
    ToolDefinition,
    ToolResult,
    make_agent_context,
)
from protocore.protocols import LLMClient
from protocore.registry import ToolRegistry
from protocore.types import ExecutionMode
from protocore.types import ToolCall


async def echo_handler(*, arguments: dict[str, Any], context: object) -> ToolResult:
    """Обработчик инструмента echo: возвращает переданный аргумент или подтверждение."""
    value = arguments.get("value", "ok")
    return ToolResult(
        tool_call_id="",  # оркестратор подставит фактический id
        tool_name="echo",
        content=str(value),
    )


def make_mock_llm_tool_then_final() -> LLMClient:
    """Мок LLM: первый вызов — tool call (echo), второй — финальный текст."""
    call_count = 0
    llm = MagicMock(spec=LLMClient)

    async def complete(*args: Any, **kwargs: Any) -> Message:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall.model_validate(
                        {
                            "id": "tc1",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value": "Echo from mock"}',
                            },
                        }
                    )
                ],
            )
        return Message(role="assistant", content="Готово. Инструмент echo вызван.")

    llm.complete = AsyncMock(side_effect=complete)
    return cast(LLMClient, llm)


async def main() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echoes back the given value for demo",
        ),
        echo_handler,
    )

    llm = make_mock_llm_tool_then_final()
    config = AgentConfig(
        name="tool-demo",
        model="mock",
        execution_mode=ExecutionMode.BYPASS,
    )
    context = make_agent_context(config=config)
    context.messages.append(
        Message(role="user", content="Вызови инструмент echo с value=test и скажи готово.")
    )

    orchestrator = AgentOrchestrator(llm_client=llm, tool_registry=registry)
    result, report = await orchestrator.run(context)

    print("Итоговый ответ:", result.content)
    print(f"Tool calls в отчёте: {report.tool_calls_total}")
    print(f"Сообщений в контексте: {len(context.messages)}")


if __name__ == "__main__":
    asyncio.run(main())
