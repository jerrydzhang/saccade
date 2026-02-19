"""Projections for interpreting event logs.

Projectors transform a flat list of TraceEvents into structured views
(tree, graph, cost summaries) based on how they interpret relations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .events import (
    CostMetrics,
    EventType,
    LatencyMetrics,
    OperationMeta,
    Relation,
    TokenMetrics,
    TraceEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class TreeNode:
    """A node in the tree projection, built from TraceEvents for a single span_id."""

    id: str
    name: str
    kind: str = "generic"
    status: str = "PENDING"
    children: list[TreeNode] = field(default_factory=list)

    chunks: list[str] = field(default_factory=list)
    streamed_text: str = ""
    output: Any = None
    error: str | None = None

    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)

    operation: OperationMeta | None = None
    response_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> TokenMetrics:
        result = self.tokens
        for child in self.children:
            result = result + child.total_tokens
        return result


@dataclass
class TreeView:
    """Tree projection using 'context' relation for parent-child hierarchy."""

    roots: list[TreeNode] = field(default_factory=list)
    nodes: dict[str, TreeNode] = field(default_factory=dict)

    @property
    def total_tokens(self) -> TokenMetrics:
        result = TokenMetrics()
        for node in self.nodes.values():
            result = result + node.tokens
        return result

    def find_by_name(self, name: str) -> TreeNode | None:
        for node in self.nodes.values():
            if node.name == name:
                return node
        return None

    def find_by_kind(self, kind: str) -> list[TreeNode]:
        return [node for node in self.nodes.values() if node.kind == kind]

    def find_failed(self) -> list[TreeNode]:
        return [node for node in self.nodes.values() if node.status == "FAILED"]

    def find_incomplete(self) -> list[TreeNode]:
        return [node for node in self.nodes.values() if node.status in ("RUNNING", "PENDING")]


def project_tree(
    events: list[TraceEvent],
    *,
    hierarchy_relation: str = Relation.CONTEXT,
) -> TreeView:
    """Project events into a tree view.

    Args:
        events: List of TraceEvents to project.
        hierarchy_relation: Relation type for parent-child links (default: "context").

    Returns:
        TreeView with roots and nodes index.

    Raises:
        ValueError: If duplicate START events found for same span_id.
    """
    if not events:
        return TreeView()

    sorted_events = sorted(events, key=lambda e: e.timestamp)

    span_data: dict[str, dict[str, Any]] = {}

    for event in sorted_events:
        span_id = event.span_id

        if span_id not in span_data:
            span_data[span_id] = {
                "id": span_id,
                "name": None,
                "kind": "generic",
                "status": "PENDING",
                "chunks": [],
                "output": None,
                "error": None,
                "tokens": TokenMetrics(),
                "cost": CostMetrics(),
                "latency": LatencyMetrics(),
                "operation": None,
                "response_id": None,
                "parent_id": None,
                "metadata": {},
            }

        data = span_data[span_id]

        if event.type == EventType.START:
            if data["status"] != "PENDING":
                msg = f"Duplicate START event for span_id '{span_id}'"
                raise ValueError(msg)
            data["status"] = "RUNNING"
            data["name"] = event.name or span_id
            if event.kind:
                data["kind"] = event.kind
            if event.inputs:
                data["metadata"]["inputs"] = event.inputs
            if event.operation:
                data["operation"] = event.operation
            if event.relations and hierarchy_relation in event.relations:
                parents = event.relations[hierarchy_relation]
                if parents:
                    data["parent_id"] = parents[0]

        elif event.type == EventType.CHUNK:
            if isinstance(event.chunk, str):
                data["chunks"].append(event.chunk)
            elif isinstance(event.chunk, dict):
                data["chunks"].append(str(event.chunk))

        elif event.type == EventType.OUTPUT:
            data["output"] = event.output

        elif event.type == EventType.SUCCESS:
            data["status"] = "COMPLETED"
            if event.tokens:
                data["tokens"] = data["tokens"] + event.tokens
            if event.cost:
                data["cost"] = data["cost"] + event.cost
            if event.latency:
                data["latency"] = event.latency
            if event.operation:
                data["operation"] = event.operation
            if event.response_id:
                data["response_id"] = event.response_id

        elif event.type == EventType.ERROR:
            data["status"] = "FAILED"
            data["error"] = event.error

        elif event.type == EventType.CANCEL:
            data["status"] = "CANCELLED"

    nodes: dict[str, TreeNode] = {}
    for span_id, data in span_data.items():
        nodes[span_id] = TreeNode(
            id=span_id,
            name=data["name"] or span_id,
            kind=data["kind"],
            status=data["status"],
            chunks=data["chunks"],
            streamed_text="".join(data["chunks"]),
            output=data["output"],
            error=data["error"],
            tokens=data["tokens"],
            cost=data["cost"],
            latency=data["latency"],
            operation=data["operation"],
            response_id=data["response_id"],
            metadata=data["metadata"],
        )

    orphans: list[tuple[str, str]] = []

    for span_id, data in span_data.items():
        parent_id = data["parent_id"]
        if parent_id:
            if parent_id in nodes:
                nodes[parent_id].children.append(nodes[span_id])
            else:
                orphans.append((span_id, parent_id))

    for child_id, missing_parent_id in orphans:
        node = nodes[child_id]
        node.metadata["orphan"] = True
        node.metadata["missing_parent"] = missing_parent_id

    children_with_existing_parent = {
        span_id
        for span_id, data in span_data.items()
        if data["parent_id"] and data["parent_id"] in nodes
    }
    root_ids = set(nodes.keys()) - children_with_existing_parent
    root_nodes = [nodes[rid] for rid in root_ids if rid in nodes]

    def get_first_timestamp(node: TreeNode) -> float:
        for event in sorted_events:
            if event.span_id == node.id:
                return event.timestamp
        return 0.0

    root_nodes.sort(key=get_first_timestamp)

    return TreeView(roots=root_nodes, nodes=nodes)


@dataclass
class GraphEdge:
    """An edge in the graph projection."""

    source_id: str
    target_id: str
    relation_type: str


@dataclass
class GraphNode:
    """A node in the graph projection."""

    id: str
    name: str
    is_ghost: bool = False
    edges_in: list[GraphEdge] = field(default_factory=list)
    edges_out: list[GraphEdge] = field(default_factory=list)


@dataclass
class GraphView:
    """Graph projection with typed edges from relations."""

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)

    def ghosts(self) -> list[GraphNode]:
        return [node for node in self.nodes.values() if node.is_ghost]

    def adjacency_list(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {}
        for edge in self.edges:
            if edge.source_id not in adj:
                adj[edge.source_id] = set()
            adj[edge.source_id].add(edge.target_id)
        return adj


def project_graph(
    events: list[TraceEvent],
    *,
    edge_relations: list[str] | None = None,
) -> GraphView:
    """Project events into a graph view.

    Args:
        events: List of TraceEvents to project.
        edge_relations: Relation types to include as edges (default: ["dataflow"]).

    Returns:
        GraphView with nodes and edges.
    """
    if edge_relations is None:
        edge_relations = [Relation.DATAFLOW]

    if not events:
        return GraphView()

    sorted_events = sorted(events, key=lambda e: e.timestamp)

    span_names: dict[str, str] = {}
    all_relations: list[tuple[str, str, str]] = []

    for event in sorted_events:
        span_id = event.span_id
        if event.name:
            span_names[span_id] = event.name
        elif span_id not in span_names:
            span_names[span_id] = span_id

        if event.relations:
            for rel_type, targets in event.relations.items():
                if rel_type in edge_relations:
                    all_relations.extend((span_id, target_id, rel_type) for target_id in targets)

    nodes: dict[str, GraphNode] = {}
    for span_id, name in span_names.items():
        nodes[span_id] = GraphNode(id=span_id, name=name)

    edges: list[GraphEdge] = []
    for source_id, target_id, rel_type in all_relations:
        if target_id not in nodes:
            nodes[target_id] = GraphNode(id=target_id, name=target_id, is_ghost=True)

        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=rel_type,
        )
        edges.append(edge)
        nodes[source_id].edges_out.append(edge)
        nodes[target_id].edges_in.append(edge)

    return GraphView(nodes=nodes, edges=edges)


@dataclass
class CostGroup:
    """Aggregated metrics for a group of spans."""

    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    count: int = 0


@dataclass
class CostView:
    """Cost aggregation projection."""

    span_count: int = 0
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    by_model: dict[str, CostGroup] = field(default_factory=dict)
    by_kind: dict[str, CostGroup] = field(default_factory=dict)
    by_name: dict[str, CostGroup] = field(default_factory=dict)


def project_cost(events: list[TraceEvent]) -> CostView:
    """Project events into a cost aggregation view.

    Args:
        events: List of TraceEvents to project.

    Returns:
        CostView with aggregated metrics by model, kind, and name.
    """
    if not events:
        return CostView()

    sorted_events = sorted(events, key=lambda e: e.timestamp)

    span_data: dict[str, dict[str, Any]] = {}

    for event in sorted_events:
        span_id = event.span_id

        if span_id not in span_data:
            span_data[span_id] = {
                "name": None,
                "kind": "generic",
                "model": None,
                "tokens": TokenMetrics(),
                "cost": CostMetrics(),
                "latency": LatencyMetrics(),
                "success_seen": False,
            }

        data = span_data[span_id]

        if event.type == EventType.START:
            data["name"] = event.name or span_id
            if event.kind:
                data["kind"] = event.kind

        elif event.type == EventType.SUCCESS:
            if not data["success_seen"]:
                data["success_seen"] = True
                if event.tokens:
                    data["tokens"] = data["tokens"] + event.tokens
                if event.cost:
                    data["cost"] = data["cost"] + event.cost
                if event.latency:
                    data["latency"] = data["latency"] + event.latency
                if event.operation and event.operation.model:
                    data["model"] = event.operation.model

    view = CostView(span_count=len(span_data))

    by_model: dict[str, CostGroup] = {}
    by_kind: dict[str, CostGroup] = {}
    by_name: dict[str, CostGroup] = {}

    for data in span_data.values():
        view.tokens = view.tokens + data["tokens"]
        view.cost = view.cost + data["cost"]
        view.latency = view.latency + data["latency"]

        model = data["model"]
        if model:
            if model not in by_model:
                by_model[model] = CostGroup()
            group = by_model[model]
            group.tokens = group.tokens + data["tokens"]
            group.cost = group.cost + data["cost"]
            group.latency = group.latency + data["latency"]
            group.count += 1

        kind = data["kind"]
        if kind not in by_kind:
            by_kind[kind] = CostGroup()
        group = by_kind[kind]
        group.tokens = group.tokens + data["tokens"]
        group.cost = group.cost + data["cost"]
        group.latency = group.latency + data["latency"]
        group.count += 1

        name = data["name"]
        if name:
            if name not in by_name:
                by_name[name] = CostGroup()
            group = by_name[name]
            group.tokens = group.tokens + data["tokens"]
            group.cost = group.cost + data["cost"]
            group.latency = group.latency + data["latency"]
            group.count += 1

    return CostView(
        span_count=view.span_count,
        tokens=view.tokens,
        cost=view.cost,
        latency=view.latency,
        by_model=by_model,
        by_kind=by_kind,
        by_name=by_name,
    )


@dataclass
class StateSnapshot:
    """A snapshot of state at a point in time."""

    state: dict[str, Any] = field(default_factory=dict)
    streamed_text: str = ""


@dataclass
class SpanStateRecord:
    """State record for a single span."""

    span_id: str
    start_time: float
    end_time: float | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: list[tuple[float, Any]] = field(default_factory=list)
    chunks: list[tuple[float, str]] = field(default_factory=list)
    parent_span_id: str | None = None


@dataclass
class StateView:
    """State reconstruction projection."""

    snapshots: list[SpanStateRecord] = field(default_factory=list)
    _records_by_span: dict[str, SpanStateRecord] = field(default_factory=dict)
    _reducer: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = field(
        default=None, repr=False
    )

    def at_time(self, timestamp: float) -> StateSnapshot:
        record = self._find_record_at_time(timestamp)
        if record is None:
            return StateSnapshot()
        return self._build_snapshot(record, timestamp)

    def at_span(self, span_id: str) -> StateSnapshot:
        if span_id not in self._records_by_span:
            return StateSnapshot()
        record = self._records_by_span[span_id]
        return self._build_snapshot(record, record.start_time)

    def _find_record_at_time(self, timestamp: float) -> SpanStateRecord | None:
        for record in self.snapshots:
            if record.start_time <= timestamp and (
                record.end_time is None or timestamp <= record.end_time
            ):
                return record
        return None

    def _build_snapshot(self, record: SpanStateRecord, timestamp: float) -> StateSnapshot:
        state: dict[str, Any] = {}

        if record.parent_span_id and record.parent_span_id in self._records_by_span:
            parent = self._records_by_span[record.parent_span_id]
            parent_snapshot = self._build_snapshot(parent, record.start_time)
            state = copy.deepcopy(parent_snapshot.state)

        if record.inputs:
            state.update(copy.deepcopy(record.inputs))
        streamed_parts: list[str] = []

        for ts, chunk in record.chunks:
            if ts <= timestamp:
                streamed_parts.append(chunk)

        for ts, output in record.outputs:
            if ts <= timestamp:
                if isinstance(output, dict):
                    new_state = copy.deepcopy(output)
                    if self._reducer is not None:
                        state = self._reducer(state, new_state)
                    else:
                        state = {**state, **new_state}
                else:
                    state["output"] = output

        return StateSnapshot(
            state=state,
            streamed_text="".join(streamed_parts),
        )


def project_state(
    events: list[TraceEvent],
    *,
    reducer: Any = None,
) -> StateView:
    """Project events into a state reconstruction view.

    Args:
        events: List of TraceEvents to project.
        reducer: Optional function to merge state (current, new) -> merged.

    Returns:
        StateView with state snapshots queryable by time or span.
    """
    if not events:
        return StateView()

    sorted_events = sorted(events, key=lambda e: e.timestamp)

    records_by_span: dict[str, SpanStateRecord] = {}

    for event in sorted_events:
        span_id = event.span_id

        if span_id not in records_by_span:
            records_by_span[span_id] = SpanStateRecord(
                span_id=span_id,
                start_time=event.timestamp,
            )

        record = records_by_span[span_id]

        if event.type == EventType.START:
            record.start_time = event.timestamp
            if event.inputs and isinstance(event.inputs, dict):
                record.inputs = dict(event.inputs)
            if event.relations and Relation.CONTEXT in event.relations:
                parents = event.relations[Relation.CONTEXT]
                if parents:
                    record.parent_span_id = parents[0]

        elif event.type == EventType.CHUNK:
            if isinstance(event.chunk, str):
                record.chunks.append((event.timestamp, event.chunk))

        elif event.type == EventType.OUTPUT:
            record.outputs.append((event.timestamp, event.output))

        elif event.type in (EventType.SUCCESS, EventType.ERROR, EventType.CANCEL):
            record.end_time = event.timestamp

    records = sorted(records_by_span.values(), key=lambda r: r.start_time)

    return StateView(
        snapshots=records,
        _records_by_span=records_by_span,
        _reducer=reducer,
    )


@dataclass
class TimelineSpan:
    """A span in the timeline projection."""

    span_id: str
    name: str
    start_ms: float
    end_ms: float | None = None
    status: str = "RUNNING"
    depth: int = 0
    clock_skew_detected: bool = False

    @property
    def duration_ms(self) -> float:
        if self.end_ms is None:
            return 0.0
        duration = self.end_ms - self.start_ms
        return max(0.0, duration)


@dataclass
class TimelineView:
    """Timeline projection for concurrency and latency analysis."""

    spans: list[TimelineSpan] = field(default_factory=list)

    @property
    def start_ms(self) -> float:
        if not self.spans:
            return 0.0
        return min(s.start_ms for s in self.spans)

    @property
    def end_ms(self) -> float:
        if not self.spans:
            return 0.0
        ends = [s.end_ms for s in self.spans if s.end_ms is not None]
        if not ends:
            return 0.0
        return max(ends)

    @property
    def total_duration_ms(self) -> float:
        return self.end_ms - self.start_ms


def project_timeline(
    events: list[TraceEvent],
    *,
    hierarchy_relation: str = Relation.CONTEXT,
) -> TimelineView:
    """Project events into a timeline view.

    Args:
        events: List of TraceEvents to project.
        hierarchy_relation: Relation type for hierarchy depth (default: "context").

    Returns:
        TimelineView with spans sorted by start time.
    """
    if not events:
        return TimelineView()

    sorted_events = sorted(events, key=lambda e: e.timestamp)

    span_data: dict[str, dict[str, Any]] = {}

    for event in sorted_events:
        span_id = event.span_id

        if span_id not in span_data:
            span_data[span_id] = {
                "span_id": span_id,
                "name": span_id,
                "start_ms": event.timestamp,
                "end_ms": None,
                "status": "RUNNING",
                "parent_id": None,
            }

        data = span_data[span_id]

        if event.type == EventType.START:
            data["start_ms"] = event.timestamp
            data["name"] = event.name or span_id
            if event.relations and hierarchy_relation in event.relations:
                parents = event.relations[hierarchy_relation]
                if parents:
                    data["parent_id"] = parents[0]

        elif event.type in (EventType.SUCCESS, EventType.ERROR, EventType.CANCEL):
            data["end_ms"] = event.timestamp
            if event.type == EventType.SUCCESS:
                data["status"] = "COMPLETED"
            elif event.type == EventType.ERROR:
                data["status"] = "FAILED"
            else:
                data["status"] = "CANCELLED"

    depth_map: dict[str, int] = {}

    def get_depth(span_id: str) -> int:
        if span_id in depth_map:
            return depth_map[span_id]
        if span_id not in span_data:
            return 0
        parent_id = span_data[span_id].get("parent_id")
        if parent_id is None or parent_id not in span_data:
            depth_map[span_id] = 0
        else:
            depth_map[span_id] = get_depth(parent_id) + 1
        return depth_map[span_id]

    spans: list[TimelineSpan] = []
    for span_id, data in span_data.items():
        start = data["start_ms"]
        end = data["end_ms"]
        clock_skew = end is not None and end < start

        spans.append(
            TimelineSpan(
                span_id=span_id,
                name=data["name"],
                start_ms=start,
                end_ms=end,
                status=data["status"],
                depth=get_depth(span_id),
                clock_skew_detected=clock_skew,
            )
        )

    spans.sort(key=lambda s: s.start_ms)

    return TimelineView(spans=spans)
