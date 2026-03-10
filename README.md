# Protocore

**Protocol-first core for small models.**  
Переиспользуемое ядро оркестрации AI-агентов для OpenAI-совместимых LLM API: предсказуемый runtime-цикл, строгие контракты, безопасные инструменты и прозрачная наблюдаемость.

> English version: [README.en.md](README.en.md)

## Почему этот проект

`Protocore` закрывает "инфраструктурный" слой агентского сервиса:

- неизменяемый orchestration loop;
- tool lifecycle и безопасный dispatch;
- маршрутизацию сабагентов;
- session/report state hooks;
- event-first observability для UI и telemetry.

Это библиотека, а не готовый SaaS-платформенный сервер: вы встраиваете ее в свой сервис/API.

## Мотивация

**От себя** — почему это ядро и чем открытая версия отличается от закрытой

### Контекст

Несколько лет я занимаюсь интеграцией нейросетей в бизнес-процессы. В большинстве сценариев нужен жёстко заданный пайплайн, где от модели не требуется самостоятельной инициативы — и такой опыт подтолкнул к вопросу: *как строить систему, которая даёт модели больше автономии, но остаётся предсказуемой и пригодной для продакшена?*

### Как появилась библиотека

Около года назад я начал более активно использовать другие библиотеки, для написания агентов для пет-проектов, однако не все мои идеи можно было реализовать в рамках этих библиотек, поэтому месяц назад я начал продумывать и разрабатывать агентское ядро, заточенное под **локальные модели небольшого размера** (условно до 80B параметров, с упором на MoE модели Qwen3-3.5 30B/35B A3B). Изучил множество существующих реализаций и собрал эту библиотеку как **свод найденных решений** — чтобы систематизировать подходы и переиспользовать их в своих проектах и в production без изобретения велосипеда каждый раз.

### Про этот репозиторий

Здесь — **открытая, «базовая» версия** ядра. Сознательно не включены:


| В открытой версии                                             | В закрытом репозитории                                                |
| ------------------------------------------------------------- | --------------------------------------------------------------------- |
| Стабильное ядро, тесты, контракты                             | Real-time метрики и дашборды                                          |
| Без встроенного API-сервера                                   | Свой API-сервер и стримы                                              |
| Без готовой персистенции состояний                            | Готовая реализация сохранения состояний                               |
| Без единого RunConfig / per-run policy overrides              | Обобщённый слой политик на каждый запуск                              |
| Без специальных мер против Prompt Injection и shell injection | Улучшенная безопасность: защита от Prompt Injection и shell injection |


Всё из правой колонки и другие идеи я развиваю в **закрытом репозитории** для коммерческих продуктов и личных pet-проектов. Публичная версия остаётся ядром, на котором можно строить свои сервисы или поверх которого — расширенную проприетарную обвязку.

**Этот репозиторий я не забрасываю:** несмотря на разработку коммерческой версии, планирую периодически обновлять открытое ядро — багфиксы, совместимость с новыми зависимостями и по возможности перенос сюда улучшений, не затрагивающих закрытую обвязку.



## Ключевые возможности

- **Immutable orchestration loop**: стабильное поведение и повторяемый lifecycle.
- **Protocol-driven архитектура**: LLM, tools, state, transport, telemetry внедряются адаптерами.
- **Безопасный tool runtime**: проверки ФС-путей, shell capability с внешним sandbox/policy.
- **Subagent orchestration**: `LEADER`, `AUTO_SELECT`, `PARALLEL`, `BYPASS`.
- **Thinking controls**: token-guardrails, selective reasoning, контролируемый stream.
- **Наблюдаемость**: `EventBus` + `ExecutionReport` для мониторинга, отладки и аудита.

## Текущий quality baseline

Несмотря на хорошее тестовое покрытие и контракты, эта версия пока находится в **экспериментальном режиме**: API и поведение могут уточняться по мере развития проекта.

Последний полный прогон тестов (локально):

- `793` tests collected
- `778 passed`, `15 skipped`
- `Total coverage: 93.86%` (при требовании `>=90%`)

Подробности и команда запуска: `docs/testing.md`.

## Установка

```bash
uv sync --extra dev
pip install .
```

Требования:

- Python `3.12+`
- Пакет: `protocore`
- Импорт: `protocore`

## Быстрый старт

```python
import asyncio

from protocore import (
    AgentConfig,
    AgentOrchestrator,
    Message,
    OpenAILLMClient,
    make_agent_context,
)


async def main() -> None:
    llm = OpenAILLMClient(
        api_key="local-token",
        base_url="http://127.0.0.1:8000/v1",
        timeout=30.0,
        default_model="Qwen/Qwen3.5-35B-A3B",
    )

    config = AgentConfig(
        name="demo-agent",
        model="Qwen/Qwen3.5-35B-A3B",
        system_prompt="Ты полезный AI-ассистент.",
    )

    context = make_agent_context(config=config)
    context.messages.append(
        Message(role="user", content="Коротко объясни, что ты умеешь.")
    )

    orchestrator = AgentOrchestrator(llm_client=llm)
    result, report = await orchestrator.run(context)
    print(result.content)
    print(f"Tokens: {report.input_tokens} in, {report.output_tokens} out")


if __name__ == "__main__":
    asyncio.run(main())
```

## Публичный API (основной surface)

```python
from protocore import (
    AgentConfig,
    AgentOrchestrator,
    OpenAILLMClient,
    ToolRegistry,
    make_agent_context,
)
```

## Подмена адаптеров

```python
from protocore import AgentOrchestrator

orchestrator = AgentOrchestrator(
    llm_client=my_llm_adapter,
    tool_executor=my_tool_executor,
    state_manager=my_state_manager,
    transport=my_transport,
    telemetry_collector=my_telemetry_collector,
)
```

`state_manager=None` — валидный stateless-режим.

## Документация

- Док-хаб: [docs/README.md](docs/README.md)
- Руководство по интеграции: [docs/guide.md](docs/guide.md)
- Архитектура: [docs/architecture.md](docs/architecture.md)
- Контракт событий: [docs/events-contract.md](docs/events-contract.md)
- Примеры: [docs/examples.md](docs/examples.md) и каталог `examples/`
- Must-have аспекты ядра: [docs/core-must-haves.md](docs/core-must-haves.md)
- Тестовое покрытие и quality evidence: [docs/testing.md](docs/testing.md)

## Проверки качества

```bash
uv sync --extra dev
uv run pytest .
```

Совместимость API проверяется в `tests/test_compatibility.py`. Дополнительно можно запускать линтер, типы и аудит безопасности вручную: `ruff`, `mypy`, `bandit`, `pip-audit` (см. `pyproject.toml`).

## Для публичного репозитория

- Лицензия: [LICENSE](LICENSE)
- Правила контрибьюта: [CONTRIBUTING.md](CONTRIBUTING.md)
- Политика безопасности: [SECURITY.md](SECURITY.md)
- Кодекс поведения: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- История изменений: [CHANGELOG.md](CHANGELOG.md)

