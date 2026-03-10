# Core must-haves in Protocore

Key building blocks that make the core production-oriented.

## 1) Immutable orchestration loop

- The runtime follows a strict loop contract: model call -> tool handling -> continuation/finalization.
- Deterministic lifecycle boundaries make behavior easier to test and reason about.
- Budget limits (`max_iterations`, `max_tool_calls`, token limits) are enforced centrally.

## 2) Protocol-driven extensibility

- External dependencies are injected through contracts (LLM, tool executor, state, transport, telemetry).
- This keeps the core vendor-agnostic and easy to embed in different environments.
- Adapter replacement allows custom infra without forking the orchestration logic.

## 3) Safe tooling model

- Tool visibility is explicit (`tool_definitions`) and separated from execution (`ToolRegistry`).
- Filesystem-aware tools are constrained by `allowed_paths`.
- Shell execution is policy-gated and designed for external sandbox runtimes.
- Approval flow supports partial completion and resume for sensitive operations.

## 4) Subagent-oriented architecture

- Built-in execution modes: `BYPASS`, `LEADER`, `AUTO_SELECT`, `PARALLEL`.
- AUTO_SELECT uses capability descriptions for routing.
- Subagents preserve isolated prompts, limits, and tool visibility while remaining traceable.

## 5) Event-first observability

- Runtime emits structured lifecycle events via `EventBus`.
- `ExecutionReport` provides run outcome, usage, warnings, tool stats, and artifacts.
- This enables dashboards, telemetry sinks, and auditing without parsing internals.

## 6) Thinking and token controls

- Thinking behavior is configurable (`thinking_profile`, `thinking_run_policy`).
- Output soft/hard token limits provide practical safety rails.
- Streaming support is integrated without requiring a separate orchestration path.

## 7) Structured output and contract safety

- Structured responses can be enforced through typed schemas.
- Contract-heavy tests validate behavior across edge cases and integration boundaries.
- The framework keeps high coverage on the core runtime surface.

## 8) Publication-ready quality baseline

- Full suite currently passes with strong coverage (see `testing.md`).
- Compatibility checks and coverage threshold are enforced via pytest (see `pyproject.toml`).
- The project is positioned as an engine/core, not a monolithic end-user app.
