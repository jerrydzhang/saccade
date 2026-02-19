# Research-Agent Implementation Plan

**Status:** Planning Complete, Ready for Phase 0
**Updated:** 2026-02-18

---

## Project Context

### Goal
Build a research agent that dogfoods the saccade telemetry library. The agent will:
- Use LiteLLM for multi-provider LLM support
- Have pluggable reasoning strategies
- Support tool calling with streaming
- Produce detailed traces via saccade

### Architecture Overview

```
saccade/                          # Workspace root
├── packages/saccade/             # Library (tracing primitives)
│   ├── primitives/               # Trace, Span, Events, Projectors
│   └── integrations/             # NEW: LiteLLM wrapper, future exporters
│
└── apps/research-agent/          # Dogfood application
    └── src/research_agent/
        ├── agent.py              # Agent class + builder
        ├── reasoning/            # Strategy implementations
        ├── tools/                # Tool registry + built-ins
        ├── memory/               # Working memory
        ├── evaluation/           # Run metrics + aggregation
        └── templates/            # Python-based presets
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Configuration | Python templates (not YAML) | True composability, type safety |
| LLM integration | Streaming-first | Better UX, real-time feedback |
| Tool registration | `@tool` decorator | Simple, auto-generates schema |
| Tracing | saccade (dogfooding) | Find pain points by using it |
| Testing | TDD + cassettes | Save API costs, reproducible |

---

## Development Principles

1. **TDD**: Write test first, then implementation
2. **Cassettes**: Record LLM responses, replay in tests (use `pytest-vcr` or `recurlib`)
3. **Phase-by-phase**: Complete each phase fully before moving to next
4. **Learn and adapt**: Each phase may reveal gaps; iterate

---

## Phase Checklist

Mark phases as complete by changing status.

- [ ] **Phase 0**: Foundation
- [ ] **Phase 1**: LiteLLM Integration (Streaming-First)
- [ ] **Phase 2**: Tool Registry
- [ ] **Phase 3**: Working Memory
- [ ] **Phase 4**: Reasoning Strategies
- [ ] **Phase 5**: Agent Core
- [ ] **Phase 6**: Built-in Tools
- [ ] **Phase 7**: Templates (Builder Pattern)
- [ ] **Phase 8**: Evaluation
- [ ] **Phase 9**: Polish

---

## Phase Details

### Phase 0: Foundation

**Goal:** Set up dependencies and verify monorepo works.

**Tasks:**
- [ ] 0.1: Add `saccade[integrations]` optional dep with LiteLLM to `packages/saccade/pyproject.toml`
- [ ] 0.2: Add pytest-vcr or similar for cassette recording to research-agent
- [ ] 0.3: Run `uv sync --all-packages` and verify

**Verification:**
```bash
uv sync --all-packages
uv run pytest packages/saccade/tests/  # Still passes
```

**Files to modify:**
- `packages/saccade/pyproject.toml`
- `apps/research-agent/pyproject.toml`

---

### Phase 1: LiteLLM Integration (Streaming-First)

**Goal:** Traced LLM calls via saccade, streaming as default.

**Test-first (write these first):**
- [ ] 1.1: Test streaming response emits CHUNK events to active Span
- [ ] 1.2: Test final response captures tokens, cost, latency
- [ ] 1.3: Test tool calling works with streaming

**Implementation:**
- [ ] 1.4: Create `saccade/integrations/__init__.py`
- [ ] 1.5: Create `saccade/integrations/litellm.py` with `TracedLiteLLM` class
- [ ] 1.6: Implement streaming (default) and non-streaming modes
- [ ] 1.7: Capture metrics on Span completion
- [ ] 1.8: Create cassettes for tests

**API:**
```python
from saccade.integrations.litellm import TracedLiteLLM

llm = TracedLiteLLM(model="claude-3.5-sonnet")

# Streaming is default
async for chunk in llm.stream(messages=[...]):
    print(chunk.delta, end="")  # Emits CHUNK to active Span

# Non-streaming if needed
response = await llm.complete(messages=[...])
```

**Files to create:**
- `packages/saccade/src/saccade/integrations/__init__.py`
- `packages/saccade/src/saccade/integrations/litellm.py`
- `packages/saccade/tests/unit/integrations/test_litellm.py`
- `packages/saccade/tests/cassettes/` (for recorded responses)

**Verification:**
```bash
uv run pytest packages/saccade/tests/unit/integrations/
uv run ty check packages/saccade/src/saccade/integrations/
```

---

### Phase 2: Tool Registry

**Goal:** Decorator-based tool registration with auto-schema generation.

**Test-first:**
- [ ] 2.1: Test `@tool` decorator extracts schema from function signature
- [ ] 2.2: Test `ToolRegistry` executes tool with args
- [ ] 2.3: Test tool execution creates saccade Span

**Implementation:**
- [ ] 2.4: Create `research_agent/tools/__init__.py`
- [ ] 2.5: Create `research_agent/tools/registry.py` with `@tool` and `ToolRegistry`
- [ ] 2.6: Generate OpenAI-compatible tool schema from type hints
- [ ] 2.7: Implement async tool execution

**API:**
```python
from research_agent.tools import tool, ToolRegistry

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    return results

registry = ToolRegistry()
registry.register(web_search)

# Get schema for LLM
schema = registry.get_schemas()

# Execute tool call
result = await registry.execute("web_search", {"query": "RAG", "max_results": 3})
```

**Files to create:**
- `apps/research-agent/src/research_agent/tools/__init__.py`
- `apps/research-agent/src/research_agent/tools/registry.py`
- `apps/research-agent/tests/unit/tools/test_registry.py`

**Verification:**
```bash
uv run pytest apps/research-agent/tests/unit/tools/
uv run ty check apps/research-agent/src/research_agent/tools/
```

---

### Phase 3: Working Memory

**Goal:** Simple message list with context window awareness.

**Test-first:**
- [ ] 3.1: Test add/remove messages
- [ ] 3.2: Test token counting (using tiktoken)
- [ ] 3.3: Test truncate to fit context window

**Implementation:**
- [ ] 3.4: Create `research_agent/memory/__init__.py`
- [ ] 3.5: Create `research_agent/memory/working.py` with `WorkingMemory`
- [ ] 3.6: Implement token counting
- [ ] 3.7: Implement truncate-oldest strategy

**API:**
```python
from research_agent.memory import WorkingMemory

memory = WorkingMemory(max_tokens=100000)
memory.add_user("What is RAG?")
memory.add_assistant("RAG stands for...")
memory.add_tool_call("web_search", {"query": "RAG"})
memory.add_tool_result("web_search", "Retrieved 5 results...")

messages = memory.to_list()  # OpenAI format
```

**Open question:** Session system for continuity across runs? Deferred to v0.2.

**Files to create:**
- `apps/research-agent/src/research_agent/memory/__init__.py`
- `apps/research-agent/src/research_agent/memory/working.py`
- `apps/research-agent/tests/unit/memory/test_working.py`

**Verification:**
```bash
uv run pytest apps/research-agent/tests/unit/memory/
uv run ty check apps/research-agent/src/research_agent/memory/
```

---

### Phase 4: Reasoning Strategies

**Goal:** Pluggable reasoning protocol.

**Test-first:**
- [ ] 4.1: Test `PassthroughStrategy` returns LLM response
- [ ] 4.2: Test strategy interface is async callable

**Implementation:**
- [ ] 4.3: Create `research_agent/reasoning/__init__.py`
- [ ] 4.4: Create `research_agent/reasoning/protocol.py` with `ReasoningStrategy` protocol
- [ ] 4.5: Create `research_agent/reasoning/passthrough.py`

**API:**
```python
from research_agent.reasoning import ReasoningStrategy, PassthroughStrategy

# Protocol
class ReasoningStrategy(Protocol):
    async def __call__(
        self,
        llm: TracedLiteLLM,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ReasoningResult:
        ...

# Default implementation
strategy = PassthroughStrategy()
result = await strategy(llm, messages, tools)
# result.content, result.tool_calls
```

**Files to create:**
- `apps/research-agent/src/research_agent/reasoning/__init__.py`
- `apps/research-agent/src/research_agent/reasoning/protocol.py`
- `apps/research-agent/src/research_agent/reasoning/passthrough.py`
- `apps/research-agent/tests/unit/reasoning/test_passthrough.py`

**Verification:**
```bash
uv run pytest apps/research-agent/tests/unit/reasoning/
uv run ty check apps/research-agent/src/research_agent/reasoning/
```

---

### Phase 5: Agent Core

**Goal:** Main agent loop with full tracing.

**Test-first:**
- [ ] 5.1: Test single-turn agent returns response
- [ ] 5.2: Test tool call triggers execution and loop continues
- [ ] 5.3: Test max_steps limit respected
- [ ] 5.4: Test error handling (LLM fail, tool fail)

**Implementation:**
- [ ] 5.5: Create `research_agent/agent.py` with `Agent` class
- [ ] 5.6: Implement agent loop: LLM → check tool calls → execute → repeat
- [ ] 5.7: Wrap each step in saccade Span
- [ ] 5.8: Return `AgentResult` with output + events
- [ ] 5.9: Create integration cassettes

**API:**
```python
from research_agent import Agent
from saccade import project_cost, project_tree

agent = Agent(
    model="claude-3.5-sonnet",
    tools=[web_search],
    max_steps=10,
)

result = await agent.run("What are recent advances in RAG?")

print(result.output)
print(project_cost(result.events).total_cost)
print(project_tree(result.events))  # decision trace
```

**Note:** This phase will likely reveal gaps. Expect iteration.

**Files to create:**
- `apps/research-agent/src/research_agent/agent.py`
- `apps/research-agent/src/research_agent/__init__.py` (exports)
- `apps/research-agent/tests/integration/test_agent.py`
- `apps/research-agent/tests/cassettes/` (agent run recordings)

**Verification:**
```bash
uv run pytest apps/research-agent/tests/integration/
uv run ty check apps/research-agent/src/research_agent/
```

---

### Phase 6: Built-in Tools

**Goal:** At least one useful tool (web search).

**Test-first:**
- [ ] 6.1: Test web_search returns results
- [ ] 6.2: Test schema is correct

**Implementation:**
- [ ] 6.3: Create `research_agent/tools/builtin/__init__.py`
- [ ] 6.4: Create `research_agent/tools/builtin/web_search.py`
- [ ] 6.5: Implement using DuckDuckGo (free, no API key)

**API:**
```python
from research_agent.tools.builtin import web_search

# Already decorated with @tool
agent = Agent(tools=[web_search])
```

**Future note:** Add provider rotation (DuckDuckGo → Tavily → SerpAPI) to prevent quota limits.

**Files to create:**
- `apps/research-agent/src/research_agent/tools/builtin/__init__.py`
- `apps/research-agent/src/research_agent/tools/builtin/web_search.py`
- `apps/research-agent/tests/unit/tools/test_web_search.py`

**Verification:**
```bash
uv run pytest apps/research-agent/tests/unit/tools/test_web_search.py
```

---

### Phase 7: Templates (Builder Pattern)

**Goal:** Fluent agent configuration.

**Test-first:**
- [ ] 7.1: Test builder produces configured agent
- [ ] 7.2: Test builder methods are chainable

**Implementation:**
- [ ] 7.3: Add `AgentBuilder` to `research_agent/agent.py`
- [ ] 7.4: Create `research_agent/templates/research.py`
- [ ] 7.5: Document template pattern

**API:**
```python
from research_agent import AgentBuilder
from research_agent.tools.builtin import web_search

agent = (
    AgentBuilder()
    .with_model("claude-3.5-sonnet")
    .with_tools([web_search])
    .with_max_steps(10)
    .build()
)

result = await agent.run("query")
```

**Files to create:**
- `apps/research-agent/src/research_agent/templates/__init__.py`
- `apps/research-agent/src/research_agent/templates/research.py`
- `apps/research-agent/tests/unit/test_builder.py`

**Verification:**
```bash
uv run pytest apps/research-agent/tests/unit/test_builder.py
```

---

### Phase 8: Evaluation

**Goal:** Compare runs and aggregate metrics.

**Test-first:**
- [ ] 8.1: Test `RunMetrics` extracts data from result
- [ ] 8.2: Test `aggregate_runs` computes statistics

**Implementation:**
- [ ] 8.3: Create `research_agent/evaluation/__init__.py`
- [ ] 8.4: Create `research_agent/evaluation/metrics.py`
- [ ] 8.5: Implement simple dataset runner

**API:**
```python
from research_agent.evaluation import RunMetrics, aggregate_runs

# After multiple runs
metrics = [
    RunMetrics.from_result(r, expected="...")
    for r in results
]
summary = aggregate_runs(metrics)
# summary.avg_cost, summary.success_rate, etc.
```

**Files to create:**
- `apps/research-agent/src/research_agent/evaluation/__init__.py`
- `apps/research-agent/src/research_agent/evaluation/metrics.py`
- `apps/research-agent/tests/unit/evaluation/test_metrics.py`

**Verification:**
```bash
uv run pytest apps/research-agent/tests/unit/evaluation/
```

---

### Phase 9: Polish

**Goal:** Production-ready.

**Tasks:**
- [ ] 9.1: Error handling review (LLM fail, tool fail, max steps, timeout)
- [ ] 9.2: Logging setup (use `logging` module)
- [ ] 9.3: Type annotations complete, `ty check` clean
- [ ] 9.4: README for research-agent
- [ ] 9.5: E2E test with real LLM (record cassette)

**Verification:**
```bash
uv run pytest apps/research-agent/tests/
uv run ty check apps/research-agent/src/research_agent/
uv run ruff check apps/research-agent/src/
```

---

## Future Considerations

Not in MVP, but noted for later:

| Feature | Notes |
|---------|-------|
| Session continuity | Working memory across runs |
| More reasoning strategies | Reflexion, Tree of Thoughts |
| Web search rotation | DuckDuckGo → Tavily → SerpAPI |
| Exporters | W&B, MLflow, OpenTelemetry |
| Framework integration | LangGraph, CrewAI via CallbackHandler |
| Persistent memory | Vector store for long-term knowledge |

---

## How to Execute a Phase

For any agent picking up this plan:

1. Read the phase details above
2. Write tests first (TDD)
3. Implement to make tests pass
4. Run verification commands
5. Update the phase checklist to mark complete
6. Commit with message: `feat(research-agent): complete phase X - <description>`

**Context needed:** This plan is self-contained. You have saccade (tracing library) and are building research-agent to dogfood it.
