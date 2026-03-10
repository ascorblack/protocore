# API Reference

## `ExecutionReport`

Основные поля:

- `loop_count`: число итераций immutable loop.
- `tool_calls_total`: общее число вызовов инструментов.
- `tool_calls_by_name`: агрегаты по имени инструмента.
- `tool_failures`: число ошибочных вызовов.
- `tool_call_details`: список `ToolCallRecord` с аргументами, latency и success/failure.
- `input_tokens`, `output_tokens`, `cached_tokens`, `reasoning_tokens`: usage-метрики.
- `subagent_runs`: сводки по дочерним запускам.
- `compression_events`: события micro/auto/manual compaction.
- `parent_tokens()`: leader-only usage без child runs.
- `child_tokens_sum()`: сумма usage дочерних прогонов.
- `total_tokens_including_subagents()`: helper с учётом child runs.

Deprecated aliases:

- `iterations` -> `loop_count`
- `tool_calls_count` -> `tool_calls_total`

## `ToolCallRecord`

Поля детального tool-call telemetry:

- `tool_call_id`
- `tool_name`
- `arguments`
- `timestamp`
- `latency_ms`
- `success`
- `status` (read-only alias: `"success"` / `"failed"`)
- `error_message`

## `Result`

Поля:

- `content`: финальный user-facing текст.
- `status`: `ExecutionStatus`.
- `artifacts`: артефакты/ссылки, собранные за run.
- `metadata`: расширяемые runtime-данные, включая `metadata["structured"]`.
- `error_details`: безопасная structured-форма ошибки для failed/timeout run.

Helpers:

- `result.to_message()` -> `Message(role="assistant", content=result.content)`
- `result.get_structured(MyModel)` -> валидированный экземпляр модели или `None`

## `CompactionSummary`

Стандартные поля summary:

- `marker`
- `completed_tasks`
- `current_goal`
- `key_decisions`
- `files_modified`
- `next_steps`

Статистика compaction:

- `original_count`
- `compacted_count`
- `tokens_saved`
- `duration_ms`
- `messages_removed` (derived: `original_count - compacted_count`)

## `ToolContext`

Поддерживаемые поля:

- `allowed_paths`
- `allow_symlinks`
- `session_id`
- `trace_id`
- `agent_id`
- `tool_call_id`
- `metadata`

Для ручных shell/tool тестов используйте `build_tool_context(...)` или
`ToolContext.for_manual_tests(...)`.
