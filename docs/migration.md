# Migration Notes

## ExecutionReport renames

Старые имена всё ещё могут встречаться в пользовательских скриптах:

- `report.iterations` -> `report.loop_count`
- `report.tool_calls_count` -> `report.tool_calls_total`

`iterations` и `tool_calls_count` оставлены как deprecated aliases, но новые
примеры и интеграции должны использовать только актуальные имена.

## Structured output

Старый паттерн:

```python
parsed = result.metadata["structured"]
model = MyModel(**parsed)
```

Новый паттерн:

```python
model = result.get_structured(MyModel)
```

## Multi-turn continuation

Старый workaround:

```python
context.messages.append(Message(role="assistant", content=result.content))
```

Новый helper:

```python
context.messages.append(result.to_message())
```

## Streaming

Для простого стриминга без инструментов теперь можно использовать:

```python
async for event in llm.stream(messages=messages):
    ...
```

Если нужны tool calls в stream-контракте, по-прежнему используйте
`llm.stream_with_tools(...)`.

## Planning

`planning_strategy` задается в `AgentOrchestrator(...)`, а не в `AgentConfig`.

## Qwen / vLLM

- `AgentConfig(enable_thinking=False)` автоматически пробрасывается в
  `extra_body.chat_template_kwargs.enable_thinking=False`.
- Системные сообщения сериализуются так, чтобы все `system` сообщения шли в
  начале provider payload, что важно для Qwen chat templates.
