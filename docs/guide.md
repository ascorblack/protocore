# Руководство по Protocore

Production-ready документация для интеграции ядра в сервис агентов. Каждый раздел соответствует отдельному аспекту ядра и может обновляться независимо.

## 1) Что делает ядро

`Protocore` реализует неизменяемый цикл агента и контракты для внешних
зависимостей: LLM, инструменты, планирование, хранение состояния и транспорт.
Ядро не включает бизнес-логику сервиса и не привязано к вендору.

## 2) Термины

- **AgentConfig**: конфигурация конкретного агента.
- **AgentContext**: контекст запуска (сообщения, идентификаторы, метаданные).
- **AgentOrchestrator**: исполнитель неизменяемого цикла.
- **ToolRegistry**: хранение определений и обработчиков инструментов.
- **Subagent**: специализированный агент, выбираемый лидером или политикой.
- **ExecutionReport**: итоговый отчет о выполнении.

## 3) Базовый поток

1. Сервис создает `AgentConfig` и `AgentContext`.
2. `AgentOrchestrator.run(context)` запускает неизменяемый цикл.
3. Внутренние события и артефакты фиксируются в `ExecutionReport`.
4. Сервис получает `Result` и `ExecutionReport`.

См. также:

- `api-reference.md` — актуальные поля `ExecutionReport`, `Result`, `CompactionSummary`.
- `migration.md` — совместимость со старыми именами и частыми миграционными ловушками.

## 4) AgentConfig: ключевые поля

- `system_prompt`: системный промпт конкретного агента.
- `description`: краткое описание возможностей агента для маршрутизации.
- `execution_mode`: `BYPASS` / `LEADER` / `AUTO_SELECT` / `PARALLEL`.
- `tool_definitions`: список инструментов, которые видит модель.
- `max_tool_calls`, `max_iterations`: budget-лимиты.
- `thinking_profile`, `thinking_run_policy`, `thinking_tokens_reserve`: управление reasoning.

## 5) Инструменты и безопасность

**Разделение ответственности:**

- `ToolRegistry` хранит `ToolDefinition` + обработчик.
- `tool_definitions` в `AgentConfig` управляет видимостью для модели.
- Если `tool_definitions` пуст, оркестратор покажет все инструменты из `ToolRegistry`.

**Безопасность ФС:**

- Для инструментов с `filesystem_access=True` выполняется валидация путей
  относительно `allowed_paths` в контексте.

## 6) Subagents и AUTO_SELECT

- Субагенты регистрируются в `AgentRegistry`.
- В `AUTO_SELECT` по умолчанию используется LLM-маршрутизация по
  `AgentConfig.description`.
- `CapabilityBasedSelectionPolicy` учитывает `confidence` и откатывается на
  fallback policy при низкой уверенности.
- У каждого субагента свой `system_prompt` и свой набор `tool_definitions`.

## 7) Планирование

- В `LEADER` режиме необходим `PlanningStrategy`.
- Базовая стратегия: `NoOpPlanningStrategy`.
- Для простых single-agent сценариев обычно выбирают
  `execution_mode=ExecutionMode.BYPASS`.
- `PlanningPolicy` позволяет включать/выключать планирование по условиям.
- `planning_strategy` передается в `AgentOrchestrator(...)`, а не в `AgentConfig`.
- `manual_compact(...)` возвращает `tuple[list[Message], CompactionSummary]`, а не только список сообщений.

Минимальная copy-paste стратегия:

```python
from protocore import NoOpPlanningStrategy

planning = NoOpPlanningStrategy()
```

Кастомная стратегия:

```python
from protocore import PlanArtifact, PlanStep, PlanningStrategy

class ReleasePlanningStrategy:
    async def build_plan(self, task, context, llm_client) -> PlanArtifact:
        _ = (context, llm_client)
        return PlanArtifact(
            plan_id="release-plan",
            raw_plan=task,
            steps=[
                PlanStep(step_id="scope", description="Collect requirements", status="pending"),
                PlanStep(step_id="draft", description="Draft release notes", status="pending"),
            ],
        )

    async def update_plan(self, plan, context, llm_client) -> PlanArtifact:
        _ = (context, llm_client)
        return plan
```

## 8) Shell capability

- Для модели это обычный tool call.
- Фактическое выполнение происходит во внешнем sandbox runtime.
- `ShellSafetyPolicy` решает: allow / deny / confirm.
- При `confirm` ран завершается со статусом `PARTIAL` и ждет approval.
- Для сессии можно задать `shell_approval_rules` (regex-правила) в
  `AgentContext.metadata`, чтобы похожие команды auto-approve без ручного подтверждения.
- Для ручных shell/tool тестов используйте `build_tool_context(...)` или
  `ToolContext.for_manual_tests(...)`, а не несуществующие поля вроде
  `tool_name` / `arguments` в `ToolContext`.

## 9) Наблюдаемость

- `ExecutionReport` содержит: статус, warnings, tool calls, usage, артефакты.
- Детализация вызовов инструментов доступна через `ExecutionReport.tool_call_details`.
- У `ExecutionReport` есть deprecated-алиасы `iterations` -> `loop_count` и `tool_calls_count` -> `tool_calls_total`.
- Для `PARALLEL` есть helpers:
  `parent_tokens()`, `child_tokens_sum()`,
  `total_tokens_including_subagents()`.
- `HookManager` и `EventBus` позволяют строить метрики/логи без форков ядра.
- Подробный event-contract для сервисов и UI: `events-contract.md`

## 10) Skills

- `SkillManager` — контракт для загрузки навыков.
- `AgentConfig.skill_set` ограничивает список доступных навыков.
- Ядро добавляет runtime tool `load_skill` и индекс навыков в `system_prompt`.

## 11) Стриминг и structured output

- Для простого стриминга можно использовать `OpenAILLMClient.stream(...)`.
- Для стриминга с инструментами используется `LLMClient.stream_with_tools(...)`.
- Для structured output — `LLMClient.complete_structured(...)` или `AgentConfig.response_format`.
- После `orchestrator.run(...)` structured payload удобнее забирать через `result.get_structured(Model)`.

## 12) Тестирование

- `FakeLLMClient` в `protocore.testing` позволяет писать unit-тесты без API.

## 13) Где смотреть примеры

- Рецепты и сценарии: `examples.md`
- Исполняемые примеры: каталог `examples/`
- Hooks: `hooks.md`
- vLLM / OpenAI-compatible setup: `vllm.md`
