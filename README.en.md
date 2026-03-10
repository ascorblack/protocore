# Protocore

**Protocol-first core for small models.**  
Reusable AI agent orchestration core for OpenAI-compatible LLM APIs: predictable runtime behavior, strict contracts, safe tools, and event-first observability.

> Russian (default) version: [README.md](README.md)

## Why this project

`Protocore` covers the infrastructure layer of an agent service:
- immutable orchestration loop;
- tool lifecycle and safe dispatch;
- subagent routing;
- session/report state hooks;
- observability for UI, telemetry, and audit.

This is a library, not a full hosted platform: you embed it into your own API/service.

## Motivation

<details>
<summary><strong>From myself</strong> — why this core exists and how the open version differs from the private one</summary>

### Context

I’ve spent several years integrating neural networks into business workflows. Most of the time the need is for a fixed pipeline where the model doesn’t take initiative — and that experience led to the question: *how to build a system that gives the model more autonomy while staying predictable and production-ready?*

### How the library came about

About a year ago I started using other agent libraries more actively for pet projects, but not all of my ideas could be implemented within them, so about a month ago I began designing and developing an agent core aimed at **small local models** (roughly up to 80B parameters, with a focus on MoE models like Qwen3-3.5 30B/35B A3B). I went through many existing implementations and built this library as a **consolidation of the solutions I found** — to systematize the approach and reuse it across my own projects and in production without reinventing the wheel each time.

### About this repository

What you see here is the **open, “base” version** of that core. I deliberately left out:

| In the open repo | In the private repo |
|------------------|---------------------|
| Stable core, tests, contracts | Real-time metrics and dashboards |
| No built-in API server | Custom API server and streams |
| No ready-made state persistence | Ready-made state persistence layer |
| No unified RunConfig / per-run policy overrides | Unified policy layer per run |
| No specific measures against Prompt Injection and shell injection | Improved security: protection against Prompt Injection and shell injection |

Everything in the right column and other ideas are developed in a **private repository** for commercial products and personal pet projects. The public version is meant to stay a core you can build your own services on — or extend with your own proprietary tooling on top.

**This repo is not abandoned:** despite working on the commercial version, I plan to update the open core periodically — bugfixes, compatibility with newer dependencies, and when possible backporting improvements that don’t depend on the private tooling.

</details>

## Core features

- **Immutable orchestration loop** for deterministic lifecycle behavior.
- **Protocol-driven architecture** for LLM, tools, state, transport, and telemetry adapters.
- **Safe tool runtime** with filesystem path validation and shell policy/sandbox support.
- **Subagent orchestration** with `LEADER`, `AUTO_SELECT`, `PARALLEL`, and `BYPASS`.
- **Thinking controls** with token guardrails and controlled streaming.
- **Observability** via `EventBus` and `ExecutionReport`.

## Current quality baseline

Despite solid test coverage and contracts, this version is still **experimental**: API and behavior may evolve as the project develops.

Latest full local test run:
- `793` tests collected
- `778 passed`, `15 skipped`
- `Total coverage: 93.86%` (required threshold is `>=90%`)

See `docs/testing.md` for details.

## Install

```bash
uv sync --extra dev
pip install .
```

Requirements:
- Python `3.12+`
- Package: `protocore`
- Import path: `protocore`

## Quick start

```python
import asyncio

from protocore import (
    AgentConfig,
    AgentOrchestrator,
    Message,
    OpenAILLMClient,
    make_agent_context,
)


async def main() -> None:
    llm = OpenAILLMClient(
        api_key="local-token",
        base_url="http://127.0.0.1:8000/v1",
        timeout=30.0,
        default_model="Qwen/Qwen3.5-35B-A3B",
    )

    config = AgentConfig(
        name="demo-agent",
        model="Qwen/Qwen3.5-35B-A3B",
        system_prompt="You are a helpful AI assistant.",
    )

    context = make_agent_context(config=config)
    context.messages.append(
        Message(role="user", content="Briefly explain your capabilities.")
    )

    orchestrator = AgentOrchestrator(llm_client=llm)
    result, report = await orchestrator.run(context)
    print(result.content)
    print(f"Tokens: {report.input_tokens} in, {report.output_tokens} out")


if __name__ == "__main__":
    asyncio.run(main())
```

## Public API (main surface)

```python
from protocore import (
    AgentConfig,
    AgentOrchestrator,
    OpenAILLMClient,
    ToolRegistry,
    make_agent_context,
)
```

## Adapter replacement

```python
from protocore import AgentOrchestrator

orchestrator = AgentOrchestrator(
    llm_client=my_llm_adapter,
    tool_executor=my_tool_executor,
    state_manager=my_state_manager,
    transport=my_transport,
    telemetry_collector=my_telemetry_collector,
)
```

`state_manager=None` is a valid stateless mode.

## Documentation

- Documentation hub: [docs/README.md](docs/README.md)
- Integration guide: [docs/guide.en.md](docs/guide.en.md)
- Architecture: [docs/architecture.md](docs/architecture.md)
- Event contract: [docs/events-contract.md](docs/events-contract.md)
- Usage examples: [docs/examples.md](docs/examples.md) and `examples/`
- Core must-have building blocks: [docs/core-must-haves.md](docs/core-must-haves.md)
- Test evidence and coverage: [docs/testing.md](docs/testing.md)

## Quality checks

```bash
uv sync --extra dev
uv run pytest .
```

API compatibility checks live in `tests/test_compatibility.py`. You can also run lint, type-check, and security audit manually: `ruff`, `mypy`, `bandit`, `pip-audit` (see `pyproject.toml`).

## Public repository files

- License: [LICENSE](LICENSE)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
