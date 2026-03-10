# Hooks

`Protocore` uses Pluggy hook specs from `protocore/hooks/specs.py`.
Hook implementations must match parameter names exactly, otherwise Pluggy raises
`PluginValidationError`.

`create_plugin_manager()` — низкоуровневая helper-функция и возвращает
`pluggy.PluginManager`. В `AgentOrchestrator` передавайте `HookManager`, а не
сырое значение из `create_plugin_manager()`.

Правильные паттерны:

```python
hooks = HookManager()
hooks.register(MyPlugin())
orchestrator = AgentOrchestrator(llm_client=llm, hook_manager=hooks)
```

```python
pm = create_plugin_manager()
pm.register(MyPlugin())
hooks = HookManager(pm=pm)
orchestrator = AgentOrchestrator(llm_client=llm, hook_manager=hooks)
```

Неправильно:

```python
pm = create_plugin_manager()
orchestrator = AgentOrchestrator(llm_client=llm, hook_manager=pm)  # raises
```

## Signature Rules

- Keep method names identical to the spec.
- Keep parameter names and order identical to the spec.
- Return values only for hooks marked `firstresult=True`.

## Lifecycle

```python
def on_session_start(self, context, report) -> None: ...
def on_session_end(self, context, report) -> None: ...
```

`on_before_run` / `on_after_run` в текущем API отсутствуют.

## LLM

```python
def pre_llm_call(self, messages, context, report) -> None: ...
def on_response_generated(self, content, context, report) -> str | None: ...
```

## Tools

```python
def on_tool_registered(self, tool) -> None: ...
def on_tool_pre_execute(self, tool_name, arguments, context, report): ...
def on_tool_post_execute(self, result, context, report) -> None: ...
def on_destructive_action(self, tool_name, arguments, context) -> bool | None: ...
```

## Compaction

```python
def on_micro_compact(self, messages, context, report) -> None: ...
def on_auto_compact(self, messages, context, report) -> None: ...
def on_manual_compact(self, messages, context, report) -> None: ...
```

## Planning And Workflow

```python
def on_plan_created(self, plan, context, report) -> None: ...
def on_workflow_start(self, workflow, context, report) -> None: ...
def on_workflow_end(self, workflow, result, report) -> None: ...
```

## Subagents

```python
def on_subagent_start(self, agent_id, envelope_payload, report) -> None: ...
def on_subagent_end(self, agent_id, result, report) -> None: ...
```

Дочерние orchestrator'ы клонируют hook plugins для изоляции runtime-состояния.
Если plugin хранит append-only observer sink в `self.events`, этот список
переиспользуется между parent/subagent clone, чтобы телеметрия не терялась в
наблюдателях.

## Errors

```python
def on_error(self, error, context, report) -> None: ...
def on_cancelled(self, context, report) -> None: ...
```

## Example Plugin

```python
from protocore.hooks.specs import hookimpl

class MetricsPlugin:
    @hookimpl
    def on_session_start(self, context, report) -> None:
        print("start", context.request_id)

    @hookimpl
    def on_subagent_end(self, agent_id, result, report) -> None:
        print("subagent", agent_id, result.status)

    @hookimpl
    def on_response_generated(self, content, context, report) -> str | None:
        return content.strip()
```

If you need the full canonical source of truth, use `protocore/hooks/specs.py`.
