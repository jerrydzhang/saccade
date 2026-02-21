# RESEARCH-AGENT - AGENTS.md

**Generated:** 2026-02-20
**Commit:** 04272c9
**Branch:** main

## OVERVIEW

Dogfood application for saccade telemetry library. Autonomous research agent with pluggable tools and streaming LLM support. **Work-in-progress** (Phase 2 of 9 complete).

## STRUCTURE

```
research-agent/
├── PLAN.md                 # 9-phase implementation plan (START HERE)
├── pyproject.toml          # Package config
├── src/research_agent/
│   ├── __init__.py         # Empty (exports TBD)
│   └── tools/
│       ├── __init__.py     # Exports: tool, ToolRegistry, ToolDefinition, ToolError
│       └── registry.py     # @tool decorator, ToolRegistry (411 lines)
└── tests/
    └── unit/tools/
        └── test_registry.py  # 113 tests (1425 lines)
```

## IMPLEMENTATION STATUS

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ Complete | Foundation (deps, monorepo) |
| 1 | ✅ Complete | LiteLLM Integration (TracedLiteLLM in saccade) |
| 2 | ✅ Complete | Tool Registry (@tool, ToolRegistry) |
| 3-9 | ⏳ Pending | Working Memory, Reasoning, Agent Core, Built-in Tools, Templates, Evaluation, Polish |

See `PLAN.md` for detailed phase breakdowns and test checklists.

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add tool | `src/research_agent/tools/registry.py` — use `@tool` decorator |
| Modify tool execution | `src/research_agent/tools/registry.py` — `ToolDefinition.execute()` |
| Change schema generation | `src/research_agent/tools/registry.py` — `_convert_schema()` |
| Add type coercion | `src/research_agent/tools/registry.py` — `_coerce_value()` |
| Tool tests | `tests/unit/tools/test_registry.py` |
| Implementation plan | `PLAN.md` |

## CODE MAP

| Symbol | Type | Role |
|--------|------|------|
| `@tool` | decorator | Wraps function into ToolDefinition with auto-schema |
| `ToolDefinition` | class | Schema generation, type coercion, sync/async execution |
| `ToolRegistry` | class | Register tools, get OpenAI schemas, execute by name |
| `ToolError` | exception | Explicit error with LLM-friendly message |

## KEY PATTERNS

### Tool Registration
```python
from research_agent.tools import tool, ToolRegistry

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    return do_search(query, max_results)

registry = ToolRegistry()
registry.register(web_search)
schemas = registry.get_schemas()  # OpenAI-compatible
```

### Tool Execution with Tracing
```python
# Tools automatically create saccade Spans with kind="tool"
result = await registry.execute("web_search", {"query": "RAG", "max_results": "3"})
# ↑ "3" coerced to int 3, wrapped in Span, errors returned as ToolError string
```

### Saccade Integration
Tools import private `_current_bus` to check for active trace context:
```python
from saccade.primitives.trace import _current_bus
bus = _current_bus.get()
if bus:
    with Span(self._name, kind="tool", inputs=bound) as span:
        ...
```

## CONVENTIONS

### Type Coercion (LLM-Friendly)
- `"3"` → `3` (str to int)
- `"true"` → `True` (str to bool)
- Missing required param → ValidationError with field name
- Invalid type → Returns error string to LLM (not exception)

### Schema Generation
- Name from function name
- Description from docstring first line
- Optional params have default values
- Supports: str, int, float, bool, list[str], dict[str, T], Literal, Optional, Pydantic models

### Error Handling
- `ToolError(message)` — explicit error, returned to LLM as string
- Other exceptions — wrapped as `Tool '{name}' failed: {e}`
- Tools never crash the agent loop

## ANTI-PATTERNS

| Avoid | Use Instead |
|-------|-------------|
| `@tool` on `*args`/`**kwargs` function | Explicit typed parameters |
| Raising exceptions in tools | Return `ToolError("message")` |
| Non-string tool return values | Convert with `str(result)` (auto-converted) |
| Direct tool function calls with LLM args | `registry.execute(name, args_dict)` |

## COMMANDS

```bash
# Testing
uv run pytest apps/research-agent/tests/              # All tests
uv run pytest apps/research-agent/tests/unit/         # Unit only
uv run pytest -m unit apps/research-agent/tests/      # By marker

# Code quality
uv run ruff check apps/research-agent/src/            # Lint
uv run ruff format apps/research-agent/src/           # Format
uv run ty check apps/research-agent/src/              # Type check
```

## NOTES

- **PLAN.md is the source of truth** for implementation progress
- Tool registry is complete with 113 passing tests
- Uses workspace dependency: `saccade = { workspace = true }`
- Pre-commit hooks run `ruff` but not `ty` for research-agent yet
- Warns if tools executed without active span (tracing degraded)
