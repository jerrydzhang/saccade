# Cadence

A tracing and observability library for AI agents with built-in metrics.

**Status:** v0.1.0 | **Tests:** 204 passing

---

## What is Cadence?

Cadence provides primitives for tracing agent execution, capturing metrics (tokens, cost, latency), and analyzing execution patterns through flexible projections.

**Core Philosophy:**
- Event-driven architecture with immutable events
- Zero-config observability via `Span` context managers
- Multiple views of the same trace (tree, graph, cost, state, timeline)
- Built-in metric tracking for tokens, cost, and latency

## Installation

```bash
pip install cadence
```

## Quick Start

### Basic Tracing

```python
from cadence import Trace, Span, project_tree

with Trace() as trace:
    with Span("agent", kind="agent") as agent:
        result = do_some_work()
        agent.set_output(result)

tree = project_tree(trace.events)
print(f"Total tokens: {tree.total_tokens.input}")
```

### Streaming with Metrics

```python
from cadence import Trace, Span, TokenMetrics, CostMetrics, project_cost

with Trace() as trace:
    with Span("llm_call", kind="llm") as llm:
        # Simulate streaming
        for chunk in ["Hello", " ", "world"]:
            llm.stream(chunk)

        llm.set_output("Hello world")
        llm.set_metrics(
            tokens=TokenMetrics(input=50, output=4),
            cost=CostMetrics(usd=0.002)
        )

cost_view = project_cost(trace.events)
print(f"Cost: ${cost_view.total_cost.usd}")
```

### Nested Spans and Relations

```python
from cadence import Trace, Span, project_tree

with Trace() as trace:
    with Span("agent", kind="agent") as agent:
        with Span("planning", kind="llm") as planning:
            planning.set_output("Plan created")

        with Span("tool_execution", kind="tool") as tool:
            tool.set_output("Tool result")

tree = project_tree(trace.events)

# Traverse tree
for root in tree.roots:
    print(f"Span: {root.name}")
    for child in root.children:
        print(f"  └─ {child.name}")
```

### Real-Time Event Streaming

```python
from cadence import Trace, Span, EventType

with Trace() as trace:
    # Subscribe to live events
    def on_event(event):
        if event.type == EventType.CHUNK:
            print(f"[STREAMING] {event.chunk}")
        elif event.type == EventType.ERROR:
            print(f"[ERROR] {event.error}")

    trace.subscribe(on_event)

    with Span("llm", kind="llm") as llm:
        for chunk in ["Hello", " world"]:
            llm.stream(chunk)
```

## Public API

### Core Classes

| Class | Description |
|-------|-------------|
| `Trace` | Entry point for tracing. Creates a `TraceBus` and manages context. |
| `Span` | Context manager for tracing operations. Emits events, tracks metrics. |
| `TraceBus` | Collects events and notifies subscribers. (Internal, accessible via `cadence.primitives`) |

### Event Types

| Type | Description |
|------|-------------|
| `TraceEvent` | Immutable record of a state change in a span. |
| `EventType` | Enum of event types: `START`, `CHUNK`, `OUTPUT`, `SUCCESS`, `ERROR`, `CANCEL` |
| `Relation` | Enum of relation types: `CONTEXT`, `DATAFLOW` |

### Metric Types

All metric types support addition (`+`) for aggregation:

| Type | Fields |
|------|--------|
| `TokenMetrics` | `input`, `output`, `reasoning`, `cached`, `cache_write` |
| `CostMetrics` | `usd` (Decimal) |
| `LatencyMetrics` | `total_ms`, `time_to_first_token_ms`, `has_clock_skew` |
| `OperationMeta` | `model`, `provider`, `host`, `kind`, `correlation_id` |

### Projectors

Transform events into different views:

| Function | Returns | Description |
|----------|----------|-------------|
| `project_tree(events)` | `TreeView` | Hierarchical tree using "context" relations |
| `project_graph(events)` | `GraphView` | Directed graph with all relations |
| `project_cost(events)` | `CostView` | Aggregated cost and token metrics |
| `project_state(events, at_timestamp)` | `StateView` | Snapshot of state at a specific time |
| `project_timeline(events)` | `TimelineView` | Chronological view with temporal grouping |

## Projector Usage Examples

### Tree View

```python
from cadence import Trace, Span, project_tree

with Trace() as trace:
    with Span("parent") as p:
        with Span("child1"):
            pass
        with Span("child2"):
            pass

tree = project_tree(trace.events)
root = tree.roots[0]

print(f"Root: {root.name}")
print(f"Children: {[c.name for c in root.children]}")
print(f"Total tokens: {tree.total_tokens.input}")
print(f"Peak context: {tree.peak_context}")
```

### Graph View

```python
from cadence import Trace, Span, Relation, project_graph

with Trace() as trace:
    with Span("a") as a:
        pass

    with Span("b") as b:
        b.relate("dataflow", a.id)

graph = project_graph(trace.events)

# Find nodes by name
node_a = graph.find_by_name("a")
node_b = graph.find_by_name("b")

# Get edges by type
dataflow_edges = graph.edges_by_type(Relation.DATAFLOW)
```

### Cost View

```python
from cadence import Trace, Span, TokenMetrics, CostMetrics, project_cost

with Trace() as trace:
    with Span("llm1") as s1:
        s1.set_metrics(
            tokens=TokenMetrics(input=100, output=20),
            cost=CostMetrics(usd=0.01)
        )

    with Span("llm2") as s2:
        s2.set_metrics(
            tokens=TokenMetrics(input=50, output=10),
            cost=CostMetrics(usd=0.005)
        )

cost = project_cost(trace.events)
print(f"Total cost: ${cost.total_cost.usd}")
print(f"Total input tokens: {cost.total_tokens.input}")
print(f"Cost per 1k input: ${cost.cost_per_1k_input}")
```

### Timeline View

```python
from cadence import Trace, Span, project_timeline

with Trace() as trace:
    with Span("a"):
        pass
    with Span("b"):
        pass

timeline = project_timeline(trace.events)

# Group by time windows (e.g., 1 second)
for bucket in timeline.by_seconds(1.0):
    print(f"Time {bucket.start_time}-{bucket.end_time}: {len(bucket.events)} events")
```

### State View (Snapshot)

```python
from cadence import Trace, Span, project_state
import time

with Trace() as trace:
    with Span("a"):
        pass

    mid_point = time.time()

    with Span("b"):
        pass

# Snapshot at mid_point - only sees "a"
state = project_state(trace.events, at_timestamp=mid_point)
print(f"Active spans at snapshot: {[s.name for s in state.active_spans]}")
```

## Advanced Usage

### Custom Relations

```python
from cadence import Trace, Span, project_graph

with Trace() as trace:
    with Span("task_a") as a:
        pass

    with Span("task_b") as b:
        # Declare dependency
        b.relate("depends_on", a.id)

graph = project_graph(trace.events)
depends_edges = graph.edges_by_type("depends_on")
```

### Span Kinds

Cadence provides constants for common span kinds:

```python
from cadence import Span, SpanKind

with Span("agent", kind=SpanKind.AGENT):
    pass

with Span("llm", kind=SpanKind.LLM):
    pass

with Span("tool", kind=SpanKind.TOOL):
    pass
```

Available kinds:
- `SpanKind.AGENT` - Agent execution
- `SpanKind.LLM` - LLM generation
- `SpanKind.TOOL` - Tool execution
- `SpanKind.EMBEDDING` - Embedding generation
- `SpanKind.RETRIEVAL` - Retrieval operation

### Error Handling

```python
from cadence import Trace, Span, EventType

with Trace() as trace:
    trace.subscribe(lambda e: print(f"{e.type}: {e.name}"))

    try:
        with Span("failing_span"):
            raise ValueError("Something went wrong")
    except ValueError:
        pass

# Events captured:
# 1. START
# 2. ERROR (with error message)
# 3. Latency included on error event
```

### Partial Results on Error

When a span fails, if `set_output()` was called before the error, the output is preserved:

```python
with Span("partial_success") as s:
    s.set_output("Partial result")
    # More work that fails
    raise ValueError("Failed")

# ERROR event is emitted, but OUTPUT event comes first
# Projection will show the partial output
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific marker groups
pytest -m unit
pytest -m integration
pytest -m e2e
```

## Documentation

- **[TYPES.md](docs/TYPES.md)** - Complete type specification
- **[DECISIONS.md](docs/DECISIONS.md)** - Architectural decisions
- **[DESIGN_VALIDATION.md](docs/DESIGN_VALIDATION.md)** - Design validation scenarios

## Design Principles

1. **Event-First**: State is derived purely from an append-only log of immutable `TraceEvent`s
2. **Generic Relations**: Spans declare relationships as `{type: [span_ids]}`. The projector interprets them
3. **Auto-Captured Context**: The "context" relation is automatically added from execution stack
4. **Synchronous Emission**: Events are emitted immediately to subscribers (no queue, no background tasks)
5. **User Manages Lifecycle**: Memory and persistence are the user's responsibility

## Limitations

| Limitation | Workaround |
|------------|------------|
| Single-process only | Use OpenTelemetry for distributed tracing |
| Not thread-safe | Use asyncio only |
| In-memory only | Export events periodically: `[e.model_dump() for e in trace.events]` |
| No built-in persistence | Serialize to JSON, save to database, etc. |

## License

MIT

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
