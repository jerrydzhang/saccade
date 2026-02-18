# CADENCE PRIMITIVES - AGENTS.md

Core tracing primitives: events, bus, spans, and projections.

## OVERVIEW

Event-driven tracing with immutable events. State derived from append-only event log.

## STRUCTURE

```
primitives/
├── __init__.py      # Exports all primitives
├── events.py        # TraceEvent, EventType, metrics (114 lines)
├── bus.py           # TraceBus event collection (45 lines)
├── trace.py         # Trace context manager (49 lines)
├── span.py          # Span lifecycle (231 lines)
└── projectors.py    # 5 projection functions (732 lines)
```

## WHERE TO LOOK

| Task | File | Key Symbols |
|------|------|-------------|
| Add metric type | `events.py` | `TokenMetrics`, `CostMetrics`, `LatencyMetrics` |
| Add event type | `events.py` | `EventType` enum |
| Modify event structure | `events.py` | `TraceEvent` |
| Change pub/sub behavior | `bus.py` | `TraceBus.emit()`, `TraceBus.subscribe()` |
| Context propagation | `trace.py` | `_current_bus` ContextVar |
| Span lifecycle hooks | `span.py` | `Span.__enter__()`, `Span.__exit__()` |
| Add new view | `projectors.py` | Follow `project_*` pattern |

## KEY PATTERNS

### Immutability
All data models: `model_config = ConfigDict(frozen=True)`

### Aggregation
All metrics implement `__add__`:
```python
def __add__(self, other: TokenMetrics) -> TokenMetrics:
    return TokenMetrics(input=self.input + other.input, ...)
```

### Context Propagation
```python
_current_bus: ContextVar[TraceBus | None] = ContextVar("_current_bus", default=None)
_current_span_id: ContextVar[str | None] = ContextVar("_current_span_id", default=None)
```

### Event Emission Flow
```
Trace.__enter__() → sets _current_bus
  Span.__enter__() → emits START, sets _current_span_id
    Span.stream() → emits CHUNK
    Span.set_output() → stores output
    Span.__exit__() → emits OUTPUT + SUCCESS/ERROR/CANCEL
  Trace.__exit__() → clears _current_bus
```

### Projector Pattern
All projectors: sort by timestamp → aggregate by span_id → build view

## PROJECTORS

| Function | Returns | Use Case |
|----------|---------|----------|
| `project_tree(events)` | TreeView | Hierarchical span tree |
| `project_graph(events)` | GraphView | Typed edge relationships |
| `project_cost(events)` | CostView | Token/cost aggregation |
| `project_state(events)` | StateView | Time-travel state queries |
| `project_timeline(events)` | TimelineView | Concurrency/latency analysis |

## NOTES

- Relations are extensible strings (not enums): `"depends_on"`, `"caused_by"` work
- Orphan nodes marked with `metadata["orphan"] = True` in tree
- Ghost nodes (referenced but no START) have `is_ghost = True` in graph
- Partial results preserved on error: OUTPUT emitted before ERROR
