"""Saccade: Tracing and observability library for AI agents."""

from .primitives import Span, Trace
from .primitives.events import (
    CostMetrics,
    EventType,
    LatencyMetrics,
    OperationMeta,
    Relation,
    TokenMetrics,
    TraceEvent,
)
from .primitives.projectors import (
    project_cost,
    project_graph,
    project_state,
    project_timeline,
    project_tree,
)

__version__ = "0.1.0"

__all__ = [
    "CostMetrics",
    "EventType",
    "LatencyMetrics",
    "OperationMeta",
    "Relation",
    "Span",
    "TokenMetrics",
    "Trace",
    "TraceEvent",
    "project_cost",
    "project_graph",
    "project_state",
    "project_timeline",
    "project_tree",
]
