# CADENCE - AGENTS.md

**Generated:** 2026-02-17
**Python:** 3.13+

## OVERVIEW

Tracing/observability library for AI agents. Event-driven architecture with immutable events, context-managed spans, and 5 projection views.

## STRUCTURE

```
cadence/
├── src/cadence/          # Main package (import as `cadence`)
│   └── primitives/       # Core: Trace, Span, events, projectors
├── tests/                # pytest (unit/integration/e2e markers)
├── scripts/              # ContextVar validation script
├── pyproject.toml        # Config (ruff, pytest, ty)
└── flake.nix             # Nix dev shell
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add new event type | `src/cadence/primitives/events.py` |
| Add new projector | `src/cadence/primitives/projectors.py` |
| Modify span behavior | `src/cadence/primitives/span.py` |
| Context management | `src/cadence/primitives/trace.py` |
| Event collection/pub-sub | `src/cadence/primitives/bus.py` |
| Public API exports | `src/cadence/__init__.py` |
| Unit tests | `tests/unit/primitives/test_*.py` |
| Integration tests | `tests/integration/test_pipeline.py` |
| E2E/usage patterns | `tests/e2e/test_dx_patterns.py` |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `Trace` | class | `primitives/trace.py` | Entry point, context manager |
| `Span` | class | `primitives/span.py` | Operation tracing, event emission |
| `TraceBus` | class | `primitives/bus.py` | Event collection, pub/sub |
| `TraceEvent` | class | `primitives/events.py` | Immutable event record |
| `EventType` | enum | `primitives/events.py` | START/CHUNK/OUTPUT/SUCCESS/ERROR/CANCEL |
| `project_tree` | fn | `primitives/projectors.py` | Events → hierarchical tree |
| `project_graph` | fn | `primitives/projectors.py` | Events → directed graph |
| `project_cost` | fn | `primitives/projectors.py` | Events → cost aggregation |
| `project_state` | fn | `primitives/projectors.py` | Events → state snapshot |
| `project_timeline` | fn | `primitives/projectors.py` | Events → temporal view |

## CONVENTIONS

### Tooling (Non-Standard)
- **Package manager**: `uv` (not pip/poetry)
- **Type checker**: `ty` (not mypy/pyright), `error-on-warning = true`
- **Linter**: `ruff` with `select = ["ALL"]`, line-length 100

### Test Relaxations (TDD-Friendly)
`tests/**/*.py` has intentional ignores:
- `PLC0415`, `I001` - imports not at top (TDD pattern)
- `SLF001` - private member access (testing internals)
- `S101` - assert usage

### Code Patterns
- All data models use `frozen=True` (immutable)
- All metrics implement `__add__` for aggregation
- IDs use `ulid.ULID()` (time-ordered random)
- Context propagation via `contextvars.ContextVar` (not thread-local)

## ANTI-PATTERNS

| Avoid | Use Instead |
|-------|-------------|
| `uv pip install` | `uv add` / `uv sync` |
| Edit pyproject.toml deps | `uv add <pkg>` / `uv remove <pkg>` |
| Manual venv activation | `uv run <cmd>` |
| `[project.optional-dependencies]` for dev tools | `[dependency-groups]` |
| mypy / pyright | `ty` |

## COMMANDS

```bash
# Dev environment
nix develop                    # Enter Nix shell (runs uv sync)

# Dependencies
uv sync                        # Install/update deps
uv add <pkg>                   # Add dependency
uv remove <pkg>                # Remove dependency

# Code quality
uv run ruff check --fix        # Lint
uv run ruff format             # Format
uv run ty                      # Type check (strict)

# Testing
uv run pytest tests/           # All tests
uv run pytest -m unit          # Unit only
uv run pytest -m integration   # Integration only
uv run pytest -m e2e           # E2E only
uv run pytest --cov=src        # With coverage
```

## NOTES

- **No CI/CD**: All checks run manually
- **Asyncio only**: Not thread-safe, use asyncio
- **In-memory only**: No built-in persistence
