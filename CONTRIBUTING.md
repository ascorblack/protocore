# Contributing to Protocore

Thanks for considering a contribution.

## Quick start

```bash
uv sync --extra dev
uv run pytest .
```

## Development checks

Run these before opening a pull request:

```bash
uv run pytest .
uv run ruff check .
uv run mypy protocore
```

Optional security checks:

```bash
uv run bandit -r protocore
uv run pip-audit
```

For coverage, live LLM tests, and integration tests (e.g. vLLm), see `docs/testing.md`.

## Pull request expectations

- Keep pull requests focused and small when possible.
- Add or update tests for behavior changes.
- Update docs when public API or runtime behavior changes.
- Keep examples runnable.

## Commit message guidance

Use concise, imperative messages:

- `add shell approval decision metadata`
- `fix auto-select fallback when confidence is low`
- `update docs for workflow usage accounting`

## Reporting issues

For bugs and feature requests, use GitHub Issues templates.
For security-sensitive reports, use the process in `SECURITY.md`.
