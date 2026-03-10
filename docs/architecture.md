# Архитектура Protocore

Документ описывает, как работает ядро оркестрации: от входного контекста до результата, отчётов и сохранения состояния.

## 0) Структура пакета и импорты

Единственный канонический пакет — `protocore`. Вся логика ядра живёт в нём;
внешние интеграции (например, OpenAI-клиент) — в подпакете `integrations`.

- **Публичный API:** импортируйте из корня пакета:
  `from protocore import AgentOrchestrator, AgentConfig, Message, OpenAILLMClient, make_agent_context, ToolRegistry, ...`
- **Типы и протоколы:** доступны там же (`AgentConfig`, `ExecutionMode`, `LLMClient`, `ToolExecutor` и т.д.).
- **Подмодули** (для продвинутой интеграции): `protocore.types`, `protocore.protocols`, `protocore.events`, `protocore.compression`, `protocore.hooks`, `protocore.integrations.llm`.

Конкретная реализация LLM (OpenAI/vLLM-совместимый клиент) вынесена в
`protocore.integrations.llm.openai_client.OpenAILLMClient`; свой адаптер можно
реализовать по протоколу `LLMClient` и передать в `AgentOrchestrator(llm_client=...)`.

## 1) Общая схема компонентов

```mermaid
flowchart LR
    A[Client Service / API Layer] --> B[AgentContext]
    B --> C[AgentOrchestrator]

    C --> D[PlanningPolicy / PlanningStrategy]
    C --> E[ExecutionPolicy]
    C --> F[ToolRegistry / ToolExecutor]
    C --> F2[ShellExecutor / ShellSafetyPolicy]
    C --> G[LLMClient]
    C --> H[StateManager]
    C --> I[Transport]
    C --> J[TelemetryCollector]

    G --> K[(LLM Provider / OpenAI-compatible API)]
    H --> L[(State Storage)]
    I --> M[(Message Bus)]

    C --> N[ExecutionReport]
    C --> O[Result]
```

## 2) Последовательность выполнения запроса

```mermaid
sequenceDiagram
    autonumber
    participant S as Service Layer
    participant O as AgentOrchestrator
    participant P as Planning Layer
    participant L as LLMClient
    participant T as ToolExecutor
    participant ST as StateManager

    S->>O: run(context)
    O->>ST: load_session_snapshot(...)
    O->>P: validate planning mode / maybe build or update plan
    P-->>O: plan artifact (optional)

    loop Immutable agent loop
        O->>L: complete(...) / complete_structured(...)
        L-->>O: assistant message (content/tool calls/usage)
        O->>O: validate token guardrails and stop conditions
        alt model requested tool
            O->>T: execute(tool_call)
            T-->>O: tool result
            O->>O: append tool message to history
        else final assistant output
            O-->>S: result candidate
        end
    end

    O->>ST: save_session_snapshot(...)
    O->>ST: save_execution_report(...)
    O-->>S: final Result + ExecutionReport
```

## 3) Режимы LLM API и fallback

```mermaid
flowchart TD
    A[AgentConfig.api_mode] --> B{resolved mode}
    B -->|responses| C[client.responses.create]
    B -->|chat_completions| D[client.chat.completions.create]

    C --> E{backend supports responses?}
    E -->|yes| F[parse responses output]
    E -->|no + fallback enabled| D
    E -->|no + fallback disabled| G[raise compatibility error]

    D --> H[parse chat output]
    F --> I[normalized Message]
    H --> I
```

## 4) Thinking-модели: selective thinking и guardrails

```mermaid
flowchart TD
    A[AgentConfig] --> B{thinking_profile set?}
    B -->|yes| C[Apply profile defaults<br/>temperature/top_p/top_k/min_p/...]
    B -->|no| D[Keep explicit values]
    C --> E{thinking_run_policy}
    D --> E
    E -->|force_on| F[enable_thinking=True<br/>if not explicitly set]
    E -->|force_off| G[enable_thinking=False<br/>if not explicitly set]
    E -->|auto| H[leave as is]

    F --> I[effective config]
    G --> I
    H --> I

    I --> J[resolve max_tokens + thinking_tokens_reserve]
    J --> K[LLM call]
    K --> L{usage.output_tokens}
    L -->|>= soft_limit| M[warning: soft_limit exceeded]
    L -->|>= hard_limit| N[warning: hard_limit exceeded]
```

## 5) Контракт безопасности инструментов ФС

```mermaid
flowchart LR
    A[Tool call from model] --> B{filesystem_access=True<br/>or fs tag}
    B -->|no| C[regular dispatch]
    B -->|yes| D[validate path args against allowed_paths]
    D -->|allowed| C
    D -->|denied| E[reject tool call + warning]
```

## 6) Что важно для интеграции

- `AgentOrchestrator` остается единой точкой входа для выполнения.
- Все внешние компоненты внедряются как адаптеры; ядро не зависит от конкретных SDK/БД/очередей.
- Shell-доступ оформлен как встроенный capability: для модели это обычный tool call, для сервиса — отдельный sandbox-aware протокол исполнения.
- Для shell `cwd` применяется path isolation через `allowed_paths`; запросы вне разрешенных путей отклоняются до вызова runtime.
- Для thinking-моделей управление переносится в конфиг (`thinking_profile`, `thinking_run_policy`,
  `thinking_tokens_reserve`, `output_token_*_limit`) без привязки к конкретному провайдеру.
- Для OpenAI-совместимых backend рекомендуется тестировать оба режима (`responses` и `chat_completions`)
  и явно включать fallback только там, где это ожидается.
- Видимость инструментов управляется через `AgentConfig.tool_definitions`:
  если список пуст, оркестратор показывает инструменты из `ToolRegistry`.
- AUTO_SELECT по умолчанию использует LLM-маршрутизацию по `AgentConfig.description`.
- Capability-based маршрутизация учитывает confidence; при низкой уверенности
  включается fallback policy вместо жёсткого выбора.
- В `PARALLEL` `ExecutionReport.input_tokens/output_tokens` могут уже включать
  child runs; для явной интерпретации используйте `parent_tokens()`,
  `child_tokens_sum()` и `total_tokens_including_subagents()`.
- `WorkflowEngine` не получает авто-агрегацию usage: engine сам заполняет
  `ExecutionReport`, при необходимости через `accumulate_usage_from_llm_calls(...)`.
- Полный перечень примеров кода и рецептов интеграции — в `docs/examples.md`.
  Запускаемые скрипты — в каталоге `examples/` (в т.ч. с мок-LLM без внешнего API).
