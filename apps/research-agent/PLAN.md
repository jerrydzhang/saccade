# Research-Agent Implementation Plan

**Status:** Phase 3 Complete, Ready for Phase 4
**Updated:** 2026-02-22

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
│   └── integrations/             # LiteLLM wrapper, future exporters
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
| Schema generation | Pydantic `create_model()` | Type coercion, validation, LLM-friendly errors |
| Tool execution | Blocking only (v0.1) | Simplicity; non-blocking deferred for sub-agents |
| Tracing | saccade (dogfooding) | Find pain points by using it |
| Testing | TDD + cassettes | Save API costs, reproducible |

---

## Development Principles

1. **TDD**: Write test first, then implementation
2. **Cassettes**: Record LLM responses, replay in tests (use `pytest-vcr`)
3. **Phase-by-phase**: Complete each phase fully before moving to next
4. **Learn and adapt**: Each phase may reveal gaps; iterate

---

## Phase Checklist

Mark phases as complete by changing status.

- [x] **Phase 0**: Foundation
- [x] **Phase 1**: LiteLLM Integration (Streaming-First)
- [x] **Phase 2**: Tool Registry
- [x] **Phase 3**: Working Memory (MVP - no auto-truncation)
- [ ] **Phase 4**: Reasoning Strategies
- [ ] **Phase 5**: Agent Core
- [ ] **Phase 6**: Built-in Tools
- [ ] **Phase 7**: Templates (Builder Pattern)
- [ ] **Phase 8**: Evaluation
- [ ] **Phase 9**: Polish

---

## Phase Details

### Phase 0: Foundation ✅ COMPLETE

**Goal:** Set up dependencies and verify monorepo works.

**Completed:**
- Added `saccade[integrations]` optional dep with LiteLLM
- Added pytest-vcr for cassette recording
- Verified monorepo with `uv sync --all-packages`

---

### Phase 1: LiteLLM Integration ✅ COMPLETE

**Goal:** Traced LLM calls via saccade, streaming as default.

**What was implemented:**

| Component | File | Description |
|-----------|------|-------------|
| `TracedLiteLLM` | `integrations/litellm/traced_llm.py` | Streaming/non-streaming LLM wrapper with auto-tracing |
| `ModelPricing` | `integrations/litellm/pricing.py` | Per-token cost configuration |
| `OpenAICompatibleProvider` | `integrations/litellm/openai_provider.py` | Custom provider for OpenAI-compatible APIs |
| `RegisterableProvider` | `integrations/litellm/pricing.py` | ABC for providers with pricing |

**API:**
```python
from saccade.integrations.litellm import TracedLiteLLM

llm = TracedLiteLLM(model="openai/gpt-4o-mini")

# Streaming (default)
async for chunk in llm.stream(messages=[...]):
    print(chunk.delta, end="")

# Non-streaming
result = await llm.complete(messages=[...])
print(result.content)
```

**Key features:**
- Auto-emits CHUNK events during streaming
- Captures tokens, cost, latency on Span completion
- Supports tool calling with streaming
- Custom providers with pricing registration
- VCR cassettes for reproducible tests

**Test coverage:** 281 tests, 95% coverage

---

### Phase 2: Tool Registry ✅ COMPLETE

**Goal:** Decorator-based tool registration with auto-schema generation, Pydantic validation, and automatic tracing.

**What was implemented:**

| Component | File | Description |
|-----------|------|-------------|
| `@tool` decorator | `tools/registry.py` | Extracts schema via Pydantic `create_model()` |
| `ToolRegistry` | `tools/registry.py` | Register, get_schemas, execute |
| `ToolDefinition` | `tools/registry.py` | Internal representation |
| `ToolError` | `tools/registry.py` | Explicit tool errors with LLM-friendly messages |

**API:**
```python
from research_agent.tools import tool, ToolRegistry, ToolError

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    results = do_search(query, max_results)
    return format_results(results)

# Direct call (preserves sync/async)
result = web_search("RAG techniques")  # sync

# Execute method (always async, dict input, returns T | ToolError)
result = await web_search.execute({"query": "RAG techniques"})

# Registry for LLM loop
registry = ToolRegistry()
registry.register(web_search)
schemas = registry.get_schemas()  # OpenAI function schemas
result = await registry.execute("web_search", {"query": "RAG"})
```

**Key behaviors:**

| Scenario | Behavior |
|----------|----------|
| Missing type annotation | Default to `str` |
| `*args` or `**kwargs` | Not supported (explicit error) |
| Function returns non-string | Convert with `str(result)` |
| Tool raises `ToolError` | Return custom message to LLM |
| Tool raises other exception | Return `"Tool '{name}' failed: {e}"` |
| Validation fails | Return LLM-friendly error for retry |
| Sync tool execution | Wrap in `asyncio.to_thread()` |
| No active Span | Still works (warns but continues) |

**Test coverage:** 113 tests

---

### Phase 3: Working Memory

**Goal:** Message storage with token-aware truncation and pluggable backends for future memory strategies.

**Key Design Decisions:**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default strategy | Sliding window | Simple, predictable, no LLM overhead |
| Token counting | LiteLLM `token_counter()` + API response caching | Exact counts from API, estimate only for new messages |
| Architecture | Pluggable `MemoryBackend` | Easy to add summarization, semantic retrieval later |
| Thread safety | Single-threaded (asyncio only) | Agent loops are sequential; matches saccade pattern |
| Persistence | Placeholder methods | Format TBD, defer to future version |
| Truncation + exact count | Reset to 0 | After truncation, next API call provides new exact count |
| Orphan tool results | Stored (no validation) | WorkingMemory is a dumb buffer; validation is agent's responsibility |
| Tool call IDs | `generate_tool_call_id()` helper | Returns `"call_<ulid>"` - optional utility |
| Context windows | Hardcoded lookup + override | Convenient defaults, explicit override available |

**Test Strategy:**

Token estimation tests mock `estimate_tokens()` for determinism:

```python
from unittest.mock import patch

@patch("research_agent.memory.working.estimate_tokens")
def test_truncation(self, mock_estimate):
    mock_estimate.side_effect = lambda model, messages: len(messages) * 10
    # Now tests are deterministic - each message = 10 tokens
```

This isolates test logic from LiteLLM's tokenizer variance.

**Token Counting Strategy:**

WorkingMemory uses a **hybrid approach** for accurate token counting:

```
┌─────────────────────────────────────────────────────────────┐
│                    TOKEN COUNT STRATEGY                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  After each LLM API call, cache the exact token count:      │
│                                                             │
│  response.usage.prompt_tokens = EXACT count of ALL          │
│  messages sent to the API (system, user, assistant, tool)   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [system] You are...              ← EXACT (from API) │   │
│  │ [user] Hello                     ← EXACT (from API) │   │
│  │ [assistant] Hi!                  ← EXACT (from API) │   │
│  │ [user] What is RAG?              ← EXACT (from API) │   │
│  │ [assistant] RAG is...            ← EXACT (from API) │   │
│  │ ─────────────────────────────────────────────────── │   │
│  │ [user] Search for X              ← ESTIMATED (new)  │   │
│  │ [tool result] Found 5 results    ← ESTIMATED (new)  │   │
│  │                                                     │   │
│  │ exact: 50K tokens (cached)                          │   │
│  │ estimated: ~500 tokens (LiteLLM)                    │   │
│  │ error: ±50 tokens (only on new messages!)           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ERROR IS BOUNDED BY NEW MESSAGES, NOT TOTAL CONTEXT        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Important:** The `prompt_tokens` from the API response represents the **entire conversation** sent to the model, not just the last message. This is cached via `add_assistant(prompt_tokens=...)` and used as the exact baseline for future estimations.

**Test-first:**

#### Core Operations Tests
- [ ] 3.1: Test `add_user()` creates user message
- [ ] 3.2: Test `add_assistant()` creates assistant message (text only)
- [ ] 3.3: Test `add_assistant()` with tool_calls creates message with tool_calls
- [ ] 3.4: Test `add_assistant()` with prompt_tokens caches exact count
- [ ] 3.5: Test `add_tool_result()` creates tool message with tool_call_id
- [ ] 3.6: Test `set_system()` sets system message
- [ ] 3.7: Test `to_list()` returns all messages in order
- [ ] 3.8: Test `to_list()` includes system message first
- [ ] 3.9: Test `clear()` removes messages but keeps system
- [ ] 3.10: Test `message_count()` returns correct count
- [ ] 3.11: Test `add()` accepts raw message dict

#### Token Counting Tests
- [ ] 3.12: Test `estimate_total_tokens()` returns exact count when no new messages
- [ ] 3.13: Test `estimate_total_tokens()` returns exact + estimated when new messages added
- [ ] 3.14: Test `add_assistant(prompt_tokens=X)` updates exact count baseline
- [ ] 3.15: Test token estimation uses LiteLLM token_counter
- [ ] 3.16: Test `fits_in_context()` uses estimated total
- [ ] 3.17: Test exact count tracks how many messages are confirmed
- [ ] 3.18: Test adding message after exact count increases only estimated portion

#### Truncation Tests (Sliding Window)
- [ ] 3.19: Test `to_list(truncate=True)` truncates to max_tokens
- [ ] 3.20: Test truncation preserves system message
- [ ] 3.21: Test truncation keeps most recent messages
- [ ] 3.22: Test truncation with `keep_recent` minimum
- [ ] 3.23: Test `to_list(truncate=False)` returns all messages
- [ ] 3.24: Test truncation resets exact count to 0 (requires new API call for exact count)

#### Tool Pair Preservation Tests (CRITICAL)
- [ ] 3.25: Test truncation doesn't split assistant tool_calls from tool results
- [ ] 3.26: Test cutoff on tool message finds its assistant message
- [ ] 3.27: Test cutoff with multiple tool results finds their shared assistant
- [ ] 3.28: Test orphan tool result (no matching tool_call) is stored (WorkingMemory is a dumb buffer)
- [ ] 3.29: Test parallel tool calls (multiple IDs) preserved together

#### Tool Call ID Helper Tests
- [ ] 3.30: Test `generate_tool_call_id()` returns string with "call_" prefix
- [ ] 3.31: Test `generate_tool_call_id()` returns unique IDs
- [ ] 3.32: Test `generate_tool_call_id()` returns time-ordered IDs (ULID-based)

#### Context Window Detection Tests
- [ ] 3.33: Test auto-detect context window from model name
- [ ] 3.34: Test explicit max_tokens overrides auto-detection
- [ ] 3.35: Test unknown model falls back to safe default (4096)
- [ ] 3.36: Test context window lookup table includes common models

#### Pluggable Backend Tests
- [ ] 3.37: Test default backend is SlidingWindowMemory
- [ ] 3.38: Test custom backend can be injected
- [ ] 3.39: Test MemoryBackend protocol methods are called

#### Edge Cases Tests
- [ ] 3.40: Test empty memory returns just system message
- [ ] 3.41: Test empty memory without system returns empty list
- [ ] 3.42: Test single message fits within context
- [ ] 3.43: Test message exactly at token limit
- [ ] 3.44: Test all messages exceed context (returns minimum viable)
- [ ] 3.45: Test assistant message with content AND tool_calls
- [ ] 3.46: Test tool result with error content

**Implementation:**
- [ ] 3.47: Create `research_agent/memory/__init__.py`
- [ ] 3.48: Create `research_agent/memory/working.py` with:
  - `MemoryBackend` protocol
  - `SlidingWindowMemory` class (default implementation)
  - `WorkingMemory` class (public API)
  - `generate_tool_call_id()` helper function
  - `estimate_tokens()` wrapper function (for test mocking)
- [ ] 3.49: Implement token counting via LiteLLM with exact/estimated split
- [ ] 3.50: Implement sliding window truncation
- [ ] 3.51: Implement tool pair preservation logic
- [ ] 3.52: Add context window lookup table

**API:**
```python
from research_agent.memory import WorkingMemory, generate_tool_call_id

# Tool call ID helper (optional - use if you need to generate IDs)
call_id = generate_tool_call_id()  # Returns "call_<ulid>" (time-ordered)

# Basic usage with model auto-detection
memory = WorkingMemory(model="gpt-4o-mini")
# ↑ max_tokens auto-detected as 128000

# Or explicit context size
memory = WorkingMemory(model="glm-5", max_tokens=200000)

# System message (always preserved)
memory.set_system("You are a research agent.")

# Conversation messages
memory.add_user("What is RAG?")
memory.add_assistant("RAG stands for Retrieval-Augmented Generation.")

# After LLM response - cache exact token count from API
response = await llm.complete(memory.to_list())
memory.add_assistant(
    content=response.content,
    tool_calls=response.tool_calls,
    prompt_tokens=response.usage.prompt_tokens,  # ← EXACT count of all messages
)

# Tool calling (IDs from LLM response)
memory.add_tool_result("call_abc123", "Retrieved 5 results...")

# Check if messages fit before sending
if memory.fits_in_context():
    response = await llm.complete(memory.to_list())

# Get messages for LLM (optionally truncated to fit context)
messages = memory.to_list()  # auto-truncates by default
messages = memory.to_list(truncate=False)  # get all

# Introspection
estimated = memory.estimate_total_tokens()  # exact + estimated new
count = memory.message_count()
```

**Token Counting Implementation:**
```python
class WorkingMemory:
    def __init__(
        self,
        model: str,
        max_tokens: int | None = None,
        keep_recent: int = 20,
        backend: MemoryBackend | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens or self._lookup_context(model)
        self.keep_recent = keep_recent
        self._backend = backend or SlidingWindowMemory()
        
        # Token tracking: exact from API + estimated for new
        self._exact_token_count: int = 0  # Cached from API response
        self._exact_up_to: int = 0        # How many messages are exact
    
    def add_assistant(
        self,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
        prompt_tokens: int | None = None,
    ) -> None:
        """Add assistant message.
        
        Args:
            content: Text content of the response
            tool_calls: List of tool call objects from LLM response
            prompt_tokens: EXACT token count from API response.usage.prompt_tokens.
                          This is the count of ALL messages sent to the LLM,
                          not just this message. Used to cache exact counts.
        """
        msg: dict = {"role": "assistant"}
        if content is not None:
            msg["content"] = content
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        self._backend.add(msg)
        
        if prompt_tokens is not None:
            self._exact_token_count = prompt_tokens
            self._exact_up_to = len(self._backend.get_all())
    
    def estimate_total_tokens(self) -> int:
        """Estimate total tokens: exact (cached) + estimated (new).
        
        After each API call, prompt_tokens is cached as the exact count.
        Only messages added AFTER the last cache are estimated.
        This keeps error bounded to new messages, not the entire context.
        """
        all_messages = self._backend.get_all()
        new_messages = all_messages[self._exact_up_to:]
        
        if not new_messages:
            return self._exact_token_count
        
        # Estimate new messages using LiteLLM
        import litellm
        estimated_new = litellm.token_counter(
            model=self.model,
            messages=new_messages
        )
        return self._exact_token_count + estimated_new
    
    def fits_in_context(self) -> bool:
        """Check if current messages fit within context window."""
        return self.estimate_total_tokens() <= self.max_tokens
```

**Tool Pair Preservation Logic:**
```python
def _find_safe_cutoff(self, cutoff: int) -> int:
    """Adjust cutoff to avoid splitting tool call/result pairs.
    
    If cutoff lands on a ToolMessage, search backward for its AIMessage
    and include it. This ensures the LLM receives complete tool exchanges.
    """
    messages = self._backend.get_all()
    if cutoff >= len(messages):
        return cutoff
    
    msg = messages[cutoff]
    if msg.get("role") != "tool":
        return cutoff
    
    # Collect tool_call_ids from consecutive tool messages at/after cutoff
    tool_ids: set[str] = set()
    i = cutoff
    while i < len(messages) and messages[i].get("role") == "tool":
        if tcid := messages[i].get("tool_call_id"):
            tool_ids.add(tcid)
        i += 1
    
    # Search backward for AIMessage with matching tool_calls
    for j in range(cutoff - 1, -1, -1):
        msg = messages[j]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("id") in tool_ids:
                    return j  # Include this assistant message
    
    return cutoff  # No match found, use original
```

**Pluggable Backend Architecture:**
```python
from typing import Protocol

class MemoryBackend(Protocol):
    """Protocol for memory storage backends."""
    
    def add(self, message: dict) -> None: ...
    def get_all(self) -> list[dict]: ...
    def get_for_context(
        self, 
        max_tokens: int,
        token_counter: Callable[[list[dict]], int],
    ) -> list[dict]: ...
    def clear(self) -> None: ...

# Future: SummarizingMemory, SemanticMemory, etc.
```

**Context Window Lookup:**
```python
CONTEXT_WINDOWS = {
    # OpenAI
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    # Anthropic
    "claude-3.5-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-haiku": 200000,
    # Zhipu AI / Z.AI
    "glm-4": 128000,
    "glm-4-plus": 128000,
    "glm-4-air": 128000,
    "glm-4-flash": 128000,
    "glm-4.5": 128000,
    "glm-4.7": 128000,
    "glm-5": 200000,
    # Default
    "_default": 4096,
}
```

**Agent Loop Integration Example:**
```python
async def agent_step(memory: WorkingMemory, llm: TracedLiteLLM, registry: ToolRegistry):
    # Check if messages fit before sending
    if not memory.fits_in_context():
        messages = memory.to_list(truncate=True)  # Truncate to fit
    else:
        messages = memory.to_list()
    
    # Call LLM
    response = await llm.complete(messages)
    
    # Add assistant response with EXACT token count from API
    # This caches the exact count for future estimations
    memory.add_assistant(
        content=response.content,
        tool_calls=response.tool_calls,
        prompt_tokens=response.usage.prompt_tokens,  # ← Key: exact count
    )
    
    # Handle tool calls if any...
```

**Files to create:**
- `apps/research-agent/src/research_agent/memory/__init__.py`
- `apps/research-agent/src/research_agent/memory/working.py`
- `apps/research-agent/tests/unit/memory/__init__.py`
- `apps/research-agent/tests/unit/memory/test_working.py`

**Verification:**
```bash
uv run pytest apps/research-agent/tests/unit/memory/
uv run ty check apps/research-agent/src/research_agent/memory/
uv run ruff check apps/research-agent/src/research_agent/memory/
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
| **Memory truncation/compaction** | Sliding window or summarization backends. Deferred until we understand real-world usage patterns. |
| Non-blocking tools | Needed for sub-agent delegation. Defer until we understand the pattern from blocking implementation. |
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
