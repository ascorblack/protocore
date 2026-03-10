# Protocore Guide

Production-ready documentation for integrating the core into an agent service. Each section is modular and can be updated independently.

## 1) What the core does

`Protocore` implements the immutable agent loop and contracts for external
components: LLM, tools, planning, state storage, and transport. The core is
vendor-agnostic and does not contain service business logic.

## 2) Terminology

- **AgentConfig**: per-agent configuration.
- **AgentContext**: run context (messages, identifiers, metadata).
- **AgentOrchestrator**: executor of the immutable loop.
- **ToolRegistry**: storage for tool definitions and handlers.
- **Subagent**: specialized agent selected by a leader or policy.
- **ExecutionReport**: final run report.

## 3) Base flow

1. The service creates `AgentConfig` and `AgentContext`.
2. `AgentOrchestrator.run(context)` executes the loop.
3. Internal events and artifacts are recorded in `ExecutionReport`.
4. The service receives `Result` and `ExecutionReport`.

## 4) AgentConfig: key fields

- `system_prompt`: per-agent system prompt.
- `description`: short capability description used for routing.
- `execution_mode`: `BYPASS` / `LEADER` / `AUTO_SELECT` / `PARALLEL`.
- `tool_definitions`: tools visible to the model.
- `max_tool_calls`, `max_iterations`: budget limits.
- `thinking_profile`, `thinking_run_policy`, `thinking_tokens_reserve`: reasoning control.

## 5) Tools and safety

**Separation of concerns:**

- `ToolRegistry` stores `ToolDefinition` and handler.
- `tool_definitions` on `AgentConfig` controls visibility for the model.
- If `tool_definitions` is empty, the orchestrator exposes tools from `ToolRegistry`.

**Filesystem safety:**

- Tools with `filesystem_access=True` are validated against `allowed_paths`.

## 6) Subagents and AUTO_SELECT

- Subagents are registered in `AgentRegistry`.
- AUTO_SELECT defaults to LLM routing using `AgentConfig.description`.
- Each subagent has its own `system_prompt` and `tool_definitions`.

## 7) Planning

- `LEADER` mode requires a `PlanningStrategy`.
- Built-in `NoOpPlanningStrategy` is available.
- `PlanningPolicy` can enable/disable planning conditionally.

## 8) Shell capability

- The model issues a normal tool call.
- Actual execution happens in an external sandbox runtime.
- `ShellSafetyPolicy` decides: allow / deny / confirm.
- `confirm` ends the run as `PARTIAL` and waits for approval.
- Session-scoped `shell_approval_rules` (regex matchers) can be set in
  `AgentContext.metadata` to auto-approve matching shell commands.

## 9) Observability

- `ExecutionReport` contains status, warnings, tool calls, usage, artifacts.
- `HookManager` and `EventBus` enable metrics/logging without forking the core.

## 10) Skills

- `SkillManager` is the contract for skill loading.
- `AgentConfig.skill_set` limits available skills per agent.
- The core adds the runtime `load_skill` tool and injects a skill index into `system_prompt`.

## 11) Streaming and structured output

- Streaming uses `LLMClient.stream_with_tools(...)`.
- Structured output uses `LLMClient.complete_structured(...)`.

## 12) Testing

- `FakeLLMClient` in `protocore.testing` enables unit tests without API calls.

## 13) Examples

- Recipes and scenarios: `examples.md`
- Runnable scripts: `examples/`
