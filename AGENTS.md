# SACCADE - AGENTS.md

**Generated:** 2026-02-17
**Python:** 3.13+

## OVERVIEW

Tracing/observability library for AI agents. Event-driven architecture with immutable events, context-managed spans, and 5 projection views.

## STRUCTURE

```
saccade/
├── pyproject.toml              # Workspace root (config only)
├── packages/
│   └── saccade/                # Library package
│       ├── src/saccade/        # Main package (import as `saccade`)
│       │   ├── primitives/     # Core: Trace, Span, events, projectors
│       │   └── integrations/   # LiteLLM wrapper, future exporters
│       ├── tests/              # pytest (unit/integration/e2e markers)
│       └── pyproject.toml      # Package config (ruff, pytest, ty)
├── apps/
│   └── research-agent/         # Dogfood project
│       ├── PLAN.md             # Implementation plan (start here!)
│       ├── src/research_agent/
│       ├── tests/
│       └── pyproject.toml
└── flake.nix                   # Nix dev shell
```

## IMPLEMENTATION PLAN

For research-agent development, see `apps/research-agent/PLAN.md`.

This plan is self-contained and can be executed phase-by-phase by any agent.

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add new event type | `packages/saccade/src/saccade/primitives/events.py` |
| Add new projector | `packages/saccade/src/saccade/primitives/projectors.py` |
| Modify span behavior | `packages/saccade/src/saccade/primitives/span.py` |
| Context management | `packages/saccade/src/saccade/primitives/trace.py` |
| Event collection/pub-sub | `packages/saccade/src/saccade/primitives/bus.py` |
| Public API exports | `packages/saccade/src/saccade/__init__.py` |
| LiteLLM integration | `packages/saccade/src/saccade/integrations/litellm.py` |
| Unit tests | `packages/saccade/tests/unit/primitives/test_*.py` |
| Integration tests | `packages/saccade/tests/integration/test_pipeline.py` |
| E2E/usage patterns | `packages/saccade/tests/e2e/test_dx_patterns.py` |
| Research agent | `apps/research-agent/src/research_agent/` |
| Research agent plan | `apps/research-agent/PLAN.md` |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `Trace` | class | `primitives/trace.py` | Entry point, context manager |
| `Span` | class | `primitives/span.py` | Operation tracing, event emission |
| `Span.current()` | classmethod | `primitives/span.py` | Get active span from context |
| `TraceBus` | class | `primitives/bus.py` | Event collection, pub/sub |
| `TraceEvent` | class | `primitives/events.py` | Immutable event record |
| `EventType` | enum | `primitives/events.py` | START/CHUNK/OUTPUT/SUCCESS/ERROR/CANCEL |
| `project_tree` | fn | `primitives/projectors.py` | Events → hierarchical tree |
| `project_graph` | fn | `primitives/projectors.py` | Events → directed graph |
| `project_cost` | fn | `primitives/projectors.py` | Events → cost aggregation |
| `project_state` | fn | `primitives/projectors.py` | Events → state snapshot |
| `project_timeline` | fn | `primitives/projectors.py` | Events → temporal view |
| `TracedLiteLLM` | class | `integrations/litellm.py` | LiteLLM wrapper with saccade spans |
| `Agent` | class | `research_agent/agent.py` | Main agent loop |
| `AgentBuilder` | class | `research_agent/agent.py` | Fluent agent configuration |
| `@tool` | deco | `research_agent/tools/registry.py` | Tool registration decorator |
| `WorkingMemory` | class | `research_agent/memory/working.py` | Message list management |

## CONVENTIONS

### Tooling (Non-Standard)
- **Package manager**: `uv` (not pip/poetry)
- **Type checker**: `ty` (not mypy/pyright), `error-on-warning = true`
- **Linter**: `ruff` with `select = ["ALL"]`, line-length 100

### Test Relaxations (TDD-Friendly)
`packages/saccade/tests/**/*.py` has intentional ignores:
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
uv sync --all-packages         # Install/update all workspace deps
uv add <pkg> --package saccade # Add dependency to saccade
uv add <pkg> --package research-agent  # Add dependency to research-agent

# Code quality
uv run ruff check --fix        # Lint
uv run ruff format             # Format
uv run ty check packages/saccade/src packages/saccade/tests  # Type check saccade
uv run ty check apps/research-agent/src apps/research-agent/tests  # Type check research-agent

# Testing (saccade)
uv run pytest packages/saccade/tests/  # All tests
uv run pytest -m unit          # Unit only
uv run pytest -m integration   # Integration only
uv run pytest -m e2e           # E2E only
uv run pytest --cov=packages/saccade/src  # With coverage

# Testing (research-agent)
uv run pytest apps/research-agent/tests/  # All tests
uv run pytest apps/research-agent/tests/unit/  # Unit only
uv run pytest apps/research-agent/tests/integration/  # Integration only

# Building
uv build --package saccade     # Build saccade for publishing
```

## NOTES

- **Monorepo**: Uses uv workspaces with saccade library + research-agent app
- **No CI/CD**: All checks run manually
- **Asyncio only**: Not thread-safe, use asyncio
- **In-memory only**: No built-in persistence
