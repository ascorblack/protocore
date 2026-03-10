# Примеры использования Protocore

Типовые сценарии интеграции ядра в сервис агентов. Исполняемые скрипты — в каталоге `examples/`.

Исполняемые примеры:
- `examples/minimal_run.py`
- `examples/tools_example.py`
- `examples/structured_output_example.py`
- `examples/auto_select.py`
Также: `examples/qwen_local.py` требует локальный OpenAI-совместимый endpoint.

## 1) Минимальный orchestration-цикл

```python
from protocore import (
    AgentConfig,
    AgentOrchestrator,
    Message,
    OpenAILLMClient,
    make_agent_context,
)

llm = OpenAILLMClient(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
    timeout=30.0,
    default_model="Qwen/Qwen3.5-35B-A3B",
)

config = AgentConfig(
    name="minimal-agent",
    model="Qwen/Qwen3.5-35B-A3B",
    system_prompt="Ты полезный инженерный ассистент.",
)

context = make_agent_context(config=config)
context.messages.append(Message(role="user", content="Напиши чеклист релиза."))

orchestrator = AgentOrchestrator(llm_client=llm)
result, report = await orchestrator.run(context)
print(result.content)
print(f"Tokens: {report.input_tokens} in, {report.output_tokens} out")
```

## 2) Жесткий режим только Chat Completions

```python
from protocore import AgentConfig, ApiMode

config = AgentConfig(
    name="chat-only-agent",
    model="Qwen/Qwen3.5-35B-A3B",
    api_mode=ApiMode.CHAT_COMPLETIONS,
)
```

## 3) Управление thinking-режимом для Qwen

```python
from protocore import AgentConfig, ThinkingProfilePreset, ThinkingRunPolicy

config = AgentConfig(
    name="worker-agent",
    model="Qwen/Qwen3.5-35B-A3B",
    thinking_profile=ThinkingProfilePreset.INSTRUCT_TOOL_WORKER,
    thinking_run_policy=ThinkingRunPolicy.FORCE_OFF,
    thinking_tokens_reserve=1024,
    output_token_soft_limit=4096,
    output_token_hard_limit=8192,
)
```

## 4) Включение fallback `responses -> chat`

```python
from protocore import OpenAILLMClient

llm = OpenAILLMClient(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
    default_model="Qwen/Qwen3.5-35B-A3B",
    allow_response_fallback_to_chat=True,
)
```

Используйте fallback избирательно, чтобы не скрывать реальные несовместимости API.

## 5) Инструменты: registry и видимость для модели

```python
from protocore import (
    AgentConfig,
    AgentOrchestrator,
    Message,
    ToolDefinition,
    ToolResult,
    make_agent_context,
)
from protocore.registry import ToolRegistry

async def my_echo(**kwargs: object) -> ToolResult:
    args = kwargs.get("arguments") or {}
    tc = args.get("tc", "unknown")
    return ToolResult(tool_call_id=tc, tool_name="echo", content="Echo done.")

registry = ToolRegistry()
registry.register(
    ToolDefinition.simple(
        name="echo",
        description="Echoes back a confirmation",
        params={"text": ("string", True, "Text to echo back")},
    ),
    my_echo,
)

# agent sees only tools defined here
config = AgentConfig(
    name="tool-agent",
    model="gpt-4o-mini",
    tool_definitions=[
        ToolDefinition.simple(
            name="echo",
            description="Echoes back a confirmation",
            params={"text": ("string", True, "Text to echo back")},
        )
    ],
)
context = make_agent_context(config=config)
context.messages.append(Message(role="user", content="Call the echo tool."))

orchestrator = AgentOrchestrator(llm_client=llm, tool_registry=registry)
result, report = await orchestrator.run(context)
```

Если `tool_definitions` пуст, оркестратор покажет инструменты из `ToolRegistry`.

## 6) AUTO_SELECT по описаниям возможностей

```python
from protocore import (
    AgentConfig,
    AgentOrchestrator,
    AgentRegistry,
    ExecutionMode,
    Message,
    make_agent_context,
)

registry = AgentRegistry()
registry.register(
    AgentConfig(
        agent_id="coder",
        name="Coder",
        description="Code generation and debugging",
        model="gpt-4o-mini",
    )
)
registry.register(
    AgentConfig(
        agent_id="writer",
        name="Writer",
        description="Release notes and documentation",
        model="gpt-4o-mini",
    )
)

leader_cfg = AgentConfig(
    name="leader",
    model="gpt-4o-mini",
    execution_mode=ExecutionMode.AUTO_SELECT,
)
context = make_agent_context(config=leader_cfg)
context.messages.append(Message(role="user", content="Write release notes"))

orchestrator = AgentOrchestrator(llm_client=llm, agent_registry=registry)
result, report = await orchestrator.run(context)
```

Если политика вернула низкую `confidence`, `CapabilityBasedSelectionPolicy`
автоматически применяет fallback policy.

## 6.1) PARALLEL с merge-policy

```python
from protocore import (
    AgentConfig,
    AgentOrchestrator,
    AgentRegistry,
    ExecutionMode,
    Message,
    ParallelExecutionPolicy,
    SubagentResult,
    SubagentStatus,
    make_agent_context,
)

class SimpleParallelPolicy:
    max_concurrency = 2
    timeout_seconds = 30.0
    cancellation_mode = "graceful"

    async def merge_results(self, results, agent_ids):
        _ = agent_ids
        summaries = [r.summary for r in results if r is not None and r.summary]
        return SubagentResult(status=SubagentStatus.SUCCESS, summary="\n".join(summaries))

registry = AgentRegistry()
registry.register(AgentConfig(agent_id="writer", model="gpt-4o-mini", execution_mode=ExecutionMode.BYPASS))
registry.register(AgentConfig(agent_id="reviewer", model="gpt-4o-mini", execution_mode=ExecutionMode.BYPASS))

leader_cfg = AgentConfig(agent_id="leader", model="gpt-4o-mini", execution_mode=ExecutionMode.PARALLEL)
ctx = make_agent_context(config=leader_cfg)
ctx.metadata["parallel_agent_ids"] = ["writer", "reviewer"]
ctx.messages.append(Message(role="user", content="Prepare release notes"))

orch = AgentOrchestrator(
    llm_client=llm,
    agent_registry=registry,
    parallel_execution_policy=SimpleParallelPolicy(),
)
result, report = await orch.run(ctx)
print(result.content)
print("parent:", report.parent_tokens())
print("children:", report.child_tokens_sum())
print("total:", report.total_tokens_including_subagents())
```

## 7) Shell capability через sandbox runtime

```python
from protocore import (
    AgentConfig,
    AgentOrchestrator,
    ShellAccessMode,
    ShellToolConfig,
    ShellToolProfile,
)

config = AgentConfig(
    name="leader-with-shell",
    shell_tool_config=ShellToolConfig(
        access_mode=ShellAccessMode.LEADER_ONLY,
        profile=ShellToolProfile.WORKSPACE_WRITE,
        allow_network=False,
        env_allowlist=["PATH"],
    ),
)

orchestrator = AgentOrchestrator(
    llm_client=my_llm,
    shell_executor=my_k8s_shell_executor,
    shell_safety_policy=my_shell_safety_policy,
)
```

Минимальная политика allow/deny/confirm:

```python
from protocore import PolicyDecision

class StrictShellSafetyPolicy:
    async def evaluate(self, request, context, capability):
        _ = (context, capability)
        cmd = request.command.lower()
        if "rm -rf" in cmd:
            return PolicyDecision.DENY
        if "git push" in cmd:
            return PolicyDecision.CONFIRM
        return PolicyDecision.ALLOW
```

Если shell-команда требует подтверждения (`PolicyDecision.CONFIRM`), ран
завершается со статусом `PARTIAL` и возвращает `pending_shell_approval` в
`Result.metadata`. После подтверждения сервис должен передать и исходный plan,
и решение:

```python
from protocore import resume_from_pending

first_result, _first_report = await orchestrator.run(context)
context.metadata.update(resume_from_pending(first_result, "approve"))
second_result, second_report = await orchestrator.run(context)
```

Если передать только `shell_approval_decisions` без
`pending_shell_approval`, orchestrator вернет `PENDING_SHELL_APPROVAL_REQUIRED`.

Сессионный allowlist для shell approval можно задать заранее или пополнять
после ручного approve. Правила — это regex-паттерны по `tool_name`, `command`,
`cwd` и `reason`:

```python
from protocore import ShellApprovalRule, AgentContextMeta

context.metadata[AgentContextMeta.SHELL_APPROVAL_RULES] = [
    ShellApprovalRule(
        tool_name_pattern=r"^shell_exec$",
        command_pattern=r"^\\s*git(?:\\s|$)",
        cwd_pattern=r"^/workspace$",
        description="Allow git read-only commands in workspace",
    ).model_dump(mode="json")
]
```

Чтобы при approve автоматически добавить похожие команды в allowlist сессии,
передайте расширенное решение:

```python
context.metadata[AgentContextMeta.SHELL_APPROVAL_DECISIONS] = {
    plan_id: {
        "decision": "approve",
        "add_to_session_allowlist": True,
    }
}
```

Для ручных shell policy тестов создавайте `ToolContext` так:

```python
from protocore import ToolContext

tool_context = ToolContext.for_manual_tests(
    agent_id="shell-tester",
    allowed_paths=["/workspace"],
    metadata={"workspace_root": "/workspace"},
)
```

## 8) Структурированный вывод (complete_structured)

```python
from pydantic import BaseModel
from protocore import Message, make_agent_context
from protocore.integrations.llm.openai_client import OpenAILLMClient

class Choice(BaseModel):
    option: str
    score: float

llm = OpenAILLMClient(api_key="...", default_model="gpt-4o-mini")
ctx = make_agent_context(config=...)
ctx.messages.append(Message(role="user", content="Pick one: A or B"))
instance = await llm.complete_structured(
    messages=ctx.messages,
    schema=Choice,
    system=ctx.config.system_prompt,
)
```

Либо на уровне оркестратора без prompt-hack:

```python
from pydantic import BaseModel
from protocore import AgentConfig, AgentOrchestrator, Message, make_agent_context

class Choice(BaseModel):
    option: str
    score: float

config = AgentConfig(agent_id="analyzer", model="gpt-4o-mini", response_format=Choice)
context = make_agent_context(config=config)
context.messages.append(Message(role="user", content="Pick one: A or B"))
result, _report = await AgentOrchestrator(llm_client=llm).run(context)
parsed = result.get_structured(Choice)
assert parsed is not None
print(parsed.option, parsed.score)
```

`ExecutionReport` при этом аккумулирует `input_tokens` / `output_tokens` и для structured path тоже.

## 9) Стриминг

```python
from protocore import Message
from protocore.integrations.llm.openai_client import OpenAILLMClient

llm = OpenAILLMClient(api_key="...", default_model="gpt-4o-mini")
messages = [Message(role="user", content="Say hello in 5 words")]

async for event in llm.stream(messages=messages):
    if event.get("type") == "delta":
        print(event.get("text", ""), end="", flush=True)
```

Если нужны tool calls в том же потоке, используйте `llm.stream_with_tools(...)`.

## 10) Text helper для `Message.content`

```python
from protocore import get_text_content

text = get_text_content(message)
```

Helper нормализует `str | list[ContentPart] | None` в строку и убирает типовые runtime checks в клиентском коде.

## 10.1) Workflow DAG (минимальный контракт)

```python
from protocore import (
    ExecutionReport,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowNode,
    accumulate_usage_from_llm_calls,
)

workflow = WorkflowDefinition(
    name="release-flow",
    nodes=[
        WorkflowNode(node_id="draft", label="Draft notes", agent_id="writer"),
        WorkflowNode(node_id="review", label="Review notes", agent_id="reviewer"),
    ],
    edges=[WorkflowEdge(from_node="draft", to_node="review")],
)

context.metadata["workflow_definition"] = workflow.model_dump(mode="json")
result, report = await orchestrator.run_workflow(workflow=workflow, context=context)
print(result.status, report.workflow_id)
```

Если вы реализуете свой `WorkflowEngine`, usage нужно заполнять в
`ExecutionReport` явно:

```python
usage = accumulate_usage_from_llm_calls([message_a, message_b, raw_usage_dict])
engine_report = ExecutionReport(input_tokens=usage[0], output_tokens=usage[1])
```

## 11) Полезные команды перед публикацией

```bash
uv sync --extra dev
uv run pytest .
```

Дополнительно: `ruff check .`, `mypy protocore`, `bandit -r protocore`, `pip-audit` (см. `pyproject.toml`).
