# vLLM Compatibility

`Protocore` works with OpenAI-compatible endpoints such as vLLM and SGLang.

## Minimal Setup

```python
from protocore import AgentConfig, ApiMode
from protocore.integrations.llm.openai_client import OpenAILLMClient

llm = OpenAILLMClient(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
    default_model="Qwen/Qwen3.5-35B-A3B",
    api_mode=ApiMode.CHAT_COMPLETIONS,
    timeout=30.0,
)

config = AgentConfig(
    agent_id="local-agent",
    model="Qwen/Qwen3.5-35B-A3B",
    api_mode=ApiMode.CHAT_COMPLETIONS,
    enable_thinking=False,
    llm_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

## Notes

- `base_url` should point to the OpenAI-compatible `/v1` root.
- `timeout` is configurable in `OpenAILLMClient(...)`.
- `timeout` controls provider/client request timeout, not full business-scenario
  wall clock. For a hard scenario timeout, wrap `orchestrator.run(...)` in
  `asyncio.wait_for(...)`.
- `llm_extra_body` is merged into each request and is the correct place for backend-specific knobs such as `top_k`, `min_p`, or `chat_template_kwargs`.
- Structured output via `AgentConfig.response_format` works through the orchestrator and now contributes to `ExecutionReport.input_tokens` / `output_tokens`.
- Auto/manual compaction path now forces `enable_thinking=False` to reduce schema-breaking reasoning prefixes before JSON.
- If your Responses API implementation is incomplete, use `api_mode=ApiMode.CHAT_COMPLETIONS`.
- For stream usage accounting on chat backends, the client requests `stream_options.include_usage=True` when the provider supports it.
- `stream_with_tools(...)` strips runtime-only kwargs such as `tool_registry`
  and `tool_definitions` before calling the provider SDK.

## Qwen Thinking Control

For Qwen-style backends:

```python
config = AgentConfig(
    agent_id="worker",
    model="Qwen/Qwen3.5-35B-A3B",
    enable_thinking=False,
    llm_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

Or use presets:

```python
from protocore import ThinkingProfilePreset, ThinkingRunPolicy

config = AgentConfig(
    agent_id="worker",
    model="Qwen/Qwen3.5-35B-A3B",
    thinking_profile=ThinkingProfilePreset.INSTRUCT_TOOL_WORKER,
    thinking_run_policy=ThinkingRunPolicy.FORCE_OFF,
)
```

## Common Failure Modes

- `response_format not supported`: switch to Chat Completions or enable fallback if your backend only partially implements Responses.
- `0/0 tokens`: fixed for orchestrator structured-output path; if this still appears, inspect provider usage payload shape.
- Duplicate provider-specific params: pass them once through `llm_request_kwargs` or `llm_extra_body`, not both.
