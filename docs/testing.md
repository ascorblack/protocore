# Testing and quality evidence

Latest full test run used as publication evidence.

## Latest run

- Date: 2026-03-10
- Command: `uv run pytest .`
- Result: `778 passed`, `15 skipped`
- Total collected: `793`
- Duration: `10.90s`
- Coverage: `93.86%` (required minimum: `90%`)

## Coverage summary

- Core orchestration and contracts are heavily covered:
  - `protocore/orchestrator.py`: `95%`
  - `protocore/types.py`: `95%`
  - `protocore/context.py`: `98%`
  - `protocore/events.py`: `96%`
  - `protocore/tool_dispatch.py`: `90%`
  - `protocore/integrations/llm/openai_client.py`: `92%`

## What this means

- The framework has broad unit/contract/integration coverage.
- Critical runtime paths (loop, tool dispatch, eventing, model adapter) are validated.
- The project currently exceeds its own quality gate threshold with margin.

## Reproduce locally

```bash
uv sync --extra dev
uv run pytest .
```

Coverage and minimum threshold are configured in `pyproject.toml` (`--cov-fail-under=90`). For additional checks (lint, type-check, security) run the dev tools directly, for example:

```bash
uv run ruff check .
uv run mypy protocore
uv run bandit -r protocore
uv run pip-audit
```

API compatibility is asserted in `tests/test_compatibility.py`.
