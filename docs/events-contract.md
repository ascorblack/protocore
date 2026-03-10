# Контракт событий Protocore

Документ описывает runtime-контракт событий для сервисов, UI, telemetry и audit-слоя, использующих ядро как фреймворк.

## Цель контракта

- дать стабильный и подробный lifecycle каждого запуска;
- сделать одинаково наблюдаемыми лидера, субагентов, workflow и compaction;
- позволить внешнему сервису строить UI и метрики без парсинга внутренних структур ядра.

## 1. Основная модель

Ядро эмитит события через `EventBus`.

У каждого события есть:

- `name`: каноническое имя вида `<domain>.<object>.<stage>`;
- `ts`: Unix timestamp в миллисекундах;
- `seq`: монотонный sequence number внутри конкретного `EventBus`;
- `run_id`: идентификатор запуска, по умолчанию совпадает с `request_id`;
- `payload`: полезная нагрузка события.

Класс события:

```python
from protocore.events import CoreEvent

event = CoreEvent(
    name="llm.call.start",
    payload={"agent_id": "leader", "request_id": "req-1", "phase": "main_turn"},
    ts=1773082509732,
    seq=42,
    run_id="req-1",
)
```

## 2. Обязательный envelope `payload`

Каждое событие, которое эмитит оркестратор, обязано содержать минимум:

- `agent_id`
- `request_id`
- `trace_id`
- `session_id`
- `parent_agent_id`
- `run_kind`
- `execution_mode`
- `phase`

Обычно также передаются:

- `scope_id`
- `iteration`
- `correlation_id`
- `attempt`
- `source_component`
- `node_id`
- `workflow_id`

Пример:

```json
{
  "name": "tool.call.start",
  "ts": 1773082509732,
  "seq": 42,
  "run_id": "9d9d7c4e",
  "payload": {
    "agent_id": "coder",
    "request_id": "9d9d7c4e",
    "trace_id": "d43f6c1b",
    "session_id": "c3b9e1f1",
    "parent_agent_id": "leader",
    "run_kind": "subagent",
    "execution_mode": "bypass",
    "phase": "tool_use",
    "scope_id": "tool:call_123",
    "iteration": 2,
    "correlation_id": "call_123",
    "source_component": "tool_dispatch"
  }
}
```

## 3. Значение служебных полей

### `run_kind`

- `leader`: верхнеуровневый запуск оркестратора
- `subagent`: запуск дочернего агента

### `execution_mode`

Берётся из `AgentConfig.execution_mode`:

- `leader`
- `bypass`
- `auto-select`
- `parallel`
- `tool_orchestrated`

### `phase`

Ядро использует согласованный набор фаз:

- `main_turn`
- `subagent`
- `planning`
- `workflow`
- `tool_use`
- `auto_compact`
- `manual_compact`
- `synthesis`
- `finalization`

### `scope_id`

`scope_id` нужен для группировки связанных событий.

Форматы, используемые ядром:

- основной turn: `turn:<request_id>:<iteration>`
- tool call: `tool:<tool_call_id>`
- subagent: `subagent:<child_request_id>`
- compaction: `compact:<request_id>:<n>`
- workflow: `workflow:<workflow_id>`
- parallel group: `parallel:<request_id>`
- session: `session:<request_id>`

## 4. Иерархия `EventBus`

### Главное правило

У субагентов теперь свой собственный `EventBus`.

Это сделано для двух целей:

- у субагента свои лимиты подписчиков и своя локальная нагрузка;
- при этом каждое событие субагента автоматически прокидывается в родительский bus без изменений.

### Что это означает на практике

- локальные подписчики субагента подписываются на дочерний bus;
- сервис может подписаться только на bus лидера и всё равно получать события дочерних запусков;
- `CoreEvent` при прокидывании не перепаковывается: `name`, `ts`, `seq`, `run_id` и `payload` сохраняются.

Пример:

```python
from protocore import AgentOrchestrator, EventBus

root_bus = EventBus()
orchestrator = AgentOrchestrator(llm_client=my_llm, event_bus=root_bus)

events = []

async def on_event(event):
    events.append((event.name, event.payload["agent_id"], event.payload["run_kind"]))

root_bus.subscribe("*", on_event)
```

Если лидер делегирует задачу субагенту, обработчик увидит события и лидера, и дочернего агента.

## 5. Канонические домены событий

Ниже перечислены основные домены, на которые сервису стоит ориентироваться.

### Session / loop

- `session.start`
- `session.end`
- `loop.iteration.start`
- `loop.iteration.end`
- `loop.cancelled`
- `loop.error`
- `loop.budget.exceeded`

### Planning

- `planning.start`
- `planning.input.prepared`
- `planning.end`
- `planning.failed`

### LLM

- `llm.request.prepared`
- `llm.call.start`
- `llm.stream.delta`
- `llm.stream.completed`
- `llm.call.end`
- `llm.call.failed`
- `llm.output.parsed`
- `llm.output.empty`

### Tools

- `tool.call.detected`
- `tool.preflight.start`
- `tool.preflight.end`
- `tool.call.start`
- `tool.dispatch.selected`
- `tool.validation.start`
- `tool.validation.end`
- `tool.approval.required`
- `tool.approval.resolved`
- `tool.execution.start`
- `tool.execution.end`
- `tool.result.ready`
- `tool.result.appended`
- `tool.call.end`
- `tool.call.failed`
- `tool.budget.exceeded`

### Subagents

- `subagent.selection.start`
- `subagent.selection.end`
- `subagent.dispatch.start`
- `subagent.start`
- `subagent.result.raw`
- `subagent.result.parse.start`
- `subagent.result.parse.end`
- `subagent.result.fallback`
- `subagent.end`

### Parallel / synthesis

- `parallel.run.start`
- `parallel.slot.acquired`
- `parallel.slot.released`
- `parallel.child.timeout`
- `parallel.child.cancelled`
- `parallel.child.failed`
- `parallel.run.end`
- `synthesis.start`
- `synthesis.input.collected`
- `synthesis.merge.start`
- `synthesis.merge.end`
- `synthesis.end`

### Compaction / compression

- `compaction.check`
- `compression.micro`
- `compaction.auto.start`
- `compaction.llm.start`
- `compaction.llm.delta`
- `compaction.llm.end`
- `compaction.summary.parse.start`
- `compaction.summary.parse.end`
- `compaction.summary.parse_failed`
- `compression.auto`
- `compaction.manual.start`
- `compression.manual`
- `compaction.end`

### Workflow

- `workflow.start`
- `workflow.end`

### Safety / skills

- `safety.destructive_action`
- `safety.injection_signal`
- `skill.index.injected`
- `skill.load.start`
- `skill.load.end`
- `skill.budget.exceeded`

## 6. Что гарантируют события по доменам

### `session.*`

Используются как рамка запуска.

`session.start` сообщает:

- модель;
- режим API;
- включён ли stream;
- лимиты `max_iterations` и `max_tool_calls`.

`session.end` сообщает:

- конечный статус;
- `stop_reason`;
- `duration_ms`;
- возможную ошибку;
- список warning-ов.

### `loop.iteration.*`

Это turn-level границы цикла.

`loop.iteration.start` содержит:

- номер итерации;
- количество сообщений перед вызовом LLM;
- уже использованные tool calls;
- оценку токенов до шага.

`loop.iteration.end` содержит:

- сколько инструментов реально было вызвано в итерации;
- какие это были инструменты;
- количество ошибок инструментов;
- оценку токенов после шага.

### `llm.*`

`llm.request.prepared` полезен для debug/telemetry.

Он сообщает:

- количество сообщений в запросе;
- размер `system_prompt`;
- preview входа;
- количество и имена видимых инструментов.

`llm.call.start` и `llm.call.end` описывают сам вызов модели.

`llm.stream.delta` приходит на каждый text/reasoning chunk и содержит:

- `kind`: `text` или `reasoning`
- `text`
- `chars`
- `delta_index`
- `provider_event_type`

`llm.stream.completed` закрывает поток и даёт суммарную статистику.

### `tool.*`

Tool lifecycle теперь многоступенчатый и пригоден для UI.

Рекомендуемая интерпретация:

- `tool.call.detected`: модель предложила вызвать инструмент;
- `tool.preflight.*`: runtime решил allow/deny/confirm;
- `tool.call.start`: ядро приняло вызов к исполнению;
- `tool.dispatch.selected`: выбран путь `registry`, `executor` или `shell`;
- `tool.validation.*`: проверены schema/path invariants;
- `tool.approval.*`: требуется ли подтверждение и чем оно закончилось;

При auto-approve по `shell_approval_rules` событие `tool.approval.resolved`
может содержать `source="shell_approval_rule"` и `rule_id`.
- `tool.execution.*`: фактическое выполнение;
- `tool.result.ready`: результат уже получен и нормализован;
- `tool.result.appended`: tool-message добавлено в историю диалога;
- `tool.call.end`: итоговая закрывающая карточка инструмента;
- `tool.call.failed`: явная ошибка lifecycle.

`tool.call.detected` всегда несёт `arguments_preview` и `arguments_json` (как пришло от
модели). Если JSON удалось распарсить в объект, дополнительно приходят
`arguments` и `argument_keys`; для shell также `shell_command`.

Для событий lifecycle вызова инструмента (`tool.call.start`, `tool.dispatch.selected`,
`tool.execution.start`, `tool.execution.end`, `tool.result.ready`, `tool.call.end`,
`tool.call.failed`) контрактом теперь гарантируются поля входного вызова:

- `arguments`: JSON-объект аргументов (с редактированием чувствительных ключей);
- `arguments_json`: та же нагрузка в сериализованном JSON-виде;
- `argument_keys`: отсортированный список ключей аргументов.

Для shell-инструмента в этих же событиях дополнительно гарантируются:

- `shell_command`: фактическая команда, переданная в вызов;
- `shell_cwd`: переданный `cwd` (если был указан).

### `subagent.*`

В ядре сохраняется правило:

- canonical LLM/tool/session события дочернего запуска идут как обычные `llm.*`, `tool.*`, `session.*`;
- отличить их можно по `run_kind="subagent"` и `parent_agent_id`;
- домен `subagent.*` используется для lifecycle делегирования и разбора результата.

### `parallel.*` и `synthesis.*`

Используйте их для отрисовки группы параллельных запусков и отдельного блока merge/synthesis.

### `compaction.*` и `compression.*`

Разделение доменов намеренное:

- `compaction.*` показывает сам lifecycle проверки и LLM-суммаризации;
- `compression.*` означает, что сжатие уже реально применено к истории сообщений.

## 7. Минимальные правила для потребителей

Если вы строите сервис поверх ядра, придерживайтесь следующих правил.

### Не выводите run-type косвенно

Не пытайтесь определять субагента по имени события. Используйте:

- `payload["run_kind"]`
- `payload["parent_agent_id"]`
- `payload["agent_id"]`

### Группируйте UI по `scope_id`

Это самый надёжный способ собрать:

- один tool call;
- один turn;
- одну compaction-операцию;
- один subagent run.

### Используйте `request_id` для одного запуска и `trace_id` для цепочки

- `request_id` идентифицирует конкретный run;
- `trace_id` связывает лидера, дочерние запуски и параллельные ветки.

### Не полагайтесь на конкретный `seq` между разными bus

`seq` монотонен внутри конкретного `EventBus`.
Для глобальной сортировки лучше использовать `(ts, run_id, seq)`.

## 8. Примеры подписки

### Подписка на все события

```python
from protocore import EventBus
from protocore.events import CoreEvent

bus = EventBus()

async def on_any_event(event: CoreEvent) -> None:
    print(event.name, event.payload["agent_id"], event.payload["phase"])

bus.subscribe("*", on_any_event)
```

### Только события tools

```python
async def on_tool_event(event):
    if event.payload["run_kind"] == "subagent":
        print("subagent tool:", event.payload["agent_id"], event.name)

bus.subscribe("tool.call.start", on_tool_event)
bus.subscribe("tool.call.end", on_tool_event)
bus.subscribe("tool.call.failed", on_tool_event)
```

### Только streaming reasoning

```python
async def on_reasoning(event):
    if event.payload.get("kind") == "reasoning":
        print(event.payload["text"])

bus.subscribe("llm.stream.delta", on_reasoning)
```

### Отдельный timeline compaction

```python
async def on_compaction(event):
    print(event.name, event.payload["scope_id"])

for name in [
    "compaction.check",
    "compaction.auto.start",
    "compaction.llm.start",
    "compaction.llm.delta",
    "compaction.llm.end",
    "compaction.summary.parse.end",
    "compression.auto",
    "compaction.end",
]:
    bus.subscribe(name, on_compaction)
```

## 9. Что НЕ гарантируется

Контракт не гарантирует:

- наличие всех доменов событий при использовании кастомных внешних стратегий;
- одинаковый `seq` у родительского и дочернего bus;
- одинаковый объём payload для всех интеграций;
- наличие provider-specific полей вне документированных ключей.

Особенно важно:

- если вы подменяете `CompressionStrategy`, built-in `compaction.llm.*` события могут не появиться;
- если вы подменяете `WorkflowEngine`, детальные node-level события должны эмититься вашей реализацией самостоятельно.

## 10. Практическая рекомендация

Если вы строите UI поверх ядра, этого набора обычно достаточно:

- рамка run: `session.start`, `session.end`
- turn timeline: `loop.iteration.start`, `loop.iteration.end`
- LLM: `llm.call.start`, `llm.stream.delta`, `llm.call.end`
- tools: `tool.call.detected`, `tool.call.start`, `tool.result.ready`, `tool.call.end`
- subagents: `subagent.dispatch.start`, `subagent.start`, `subagent.end`
- parallel: `parallel.run.start`, `synthesis.merge.start`, `synthesis.merge.end`, `parallel.run.end`
- compaction: `compaction.auto.start`, `compaction.llm.delta`, `compression.auto`, `compaction.end`

Для аудита и продакшн-метрик дополнительно полезны:

- `tool.approval.required`
- `tool.call.failed`
- `loop.error`
- `safety.destructive_action`
- `safety.injection_signal`
