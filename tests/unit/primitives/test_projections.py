"""Tests for projections: tree, state, timeline, graph, cost."""

from decimal import Decimal

import pytest

pytestmark = pytest.mark.unit


class TestRelation:
    """Tests for built-in relation constants."""

    def test_context_relation(self):
        """Relation.CONTEXT should be defined for hierarchy."""
        from saccade.primitives.events import Relation

        assert Relation.CONTEXT == "context"

    def test_dataflow_relation(self):
        """Relation.DATAFLOW should be defined for data movement."""
        from saccade.primitives.events import Relation

        assert Relation.DATAFLOW == "dataflow"

    def test_custom_relation_allowed(self):
        """Custom relations should be usable alongside built-ins."""
        from saccade.primitives.events import EventType, TraceEvent

        event = TraceEvent(
            type=EventType.START,
            span_id="s1",
            relations={"custom_relation": ["s0"]},
        )
        assert event.relations is not None
        assert event.relations["custom_relation"] == ["s0"]


class TestProjectTree:
    """Tests for project_tree - hierarchy and status projection."""

    def test_empty_events(self):
        """project_tree should handle empty events list."""
        from saccade.primitives.projectors import TreeView, project_tree

        tree = project_tree([])
        assert isinstance(tree, TreeView)
        assert len(tree.roots) == 0
        assert len(tree.nodes) == 0

    def test_single_span(self):
        """project_tree should create a single root for a single span."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="span-1", name="root", kind="agent"
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="span-1"),
        ]

        tree = project_tree(events)
        assert len(tree.roots) == 1
        assert tree.roots[0].name == "root"
        assert tree.roots[0].status == "COMPLETED"

    def test_uses_context_by_default(self):
        """project_tree should use Relation.CONTEXT for hierarchy by default."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="agent", name="agent", kind="agent"
            ),
            TraceEvent(
                type=EventType.START,
                span_id="tool",
                name="tool",
                kind="tool",
                relations={Relation.CONTEXT: ["agent"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="tool"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="agent"),
        ]

        tree = project_tree(events)
        assert len(tree.roots) == 1
        assert tree.roots[0].name == "agent"
        assert len(tree.roots[0].children) == 1
        assert tree.roots[0].children[0].name == "tool"

    def test_configurable_hierarchy_relation(self):
        """project_tree should accept custom hierarchy relation."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="parent", name="parent"),
            TraceEvent(
                type=EventType.START,
                span_id="child",
                name="child",
                relations={"parent_span": ["parent"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="child"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="parent"),
        ]

        tree = project_tree(events, hierarchy_relation="parent_span")
        assert len(tree.roots) == 1
        assert tree.roots[0].name == "parent"
        assert len(tree.roots[0].children) == 1

    def test_deep_nesting(self):
        """project_tree should handle deep nesting."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="agent", name="agent"),
            TraceEvent(
                type=EventType.START,
                span_id="tool",
                name="tool",
                relations={Relation.CONTEXT: ["agent"]},
            ),
            TraceEvent(
                type=EventType.START,
                span_id="subtool",
                name="subtool",
                relations={Relation.CONTEXT: ["tool"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="subtool"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="tool"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="agent"),
        ]

        tree = project_tree(events)
        assert len(tree.roots) == 1
        agent = tree.roots[0]
        tool = agent.children[0]
        subtool = tool.children[0]
        assert agent.name == "agent"
        assert tool.name == "tool"
        assert subtool.name == "subtool"

    def test_multiple_roots(self):
        """project_tree should handle multiple root spans."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="agent1", name="agent1"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="agent1"),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="agent2", name="agent2"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="agent2"),
        ]

        tree = project_tree(events)
        assert len(tree.roots) == 2

    def test_ignores_dataflow_relations(self):
        """project_tree should only use hierarchy relation for structure."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="search", name="search"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="search"),
            TraceEvent(
                type=EventType.START,
                span_id="generate",
                name="generate",
                relations={Relation.DATAFLOW: ["search"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="generate"),
        ]

        tree = project_tree(events)
        assert len(tree.roots) == 2
        assert len(tree.roots[0].children) == 0

    def test_chunks_accumulated(self):
        """project_tree should accumulate streaming chunks."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="streaming"),
            TraceEvent(type=EventType.CHUNK, trace_id="t1", span_id="s1", chunk="Hello"),
            TraceEvent(type=EventType.CHUNK, trace_id="t1", span_id="s1", chunk=" "),
            TraceEvent(type=EventType.CHUNK, trace_id="t1", span_id="s1", chunk="world"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1"),
        ]

        tree = project_tree(events)
        assert tree.roots[0].chunks == ["Hello", " ", "world"]
        assert tree.roots[0].streamed_text == "Hello world"

    def test_metrics_accumulated(self):
        """project_tree should accumulate metrics from SUCCESS event."""
        from saccade.primitives.events import (
            CostMetrics,
            EventType,
            LatencyMetrics,
            TokenMetrics,
            TraceEvent,
        )
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="llm"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s1",
                tokens=TokenMetrics(input=100, output=50),
                cost=CostMetrics(usd=Decimal("0.01")),
                latency=LatencyMetrics(total_ms=100.0),
            ),
        ]

        tree = project_tree(events)
        node = tree.roots[0]
        assert node.tokens.input == 100
        assert node.tokens.output == 50
        assert node.cost.usd == Decimal("0.01")
        assert node.latency.total_ms == 100.0

    def test_response_id_from_success_event(self):
        """project_tree should capture response_id from SUCCESS event."""
        from saccade.primitives.events import EventType, OperationMeta, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(
                type=EventType.START,
                span_id="s1",
                name="call",
                operation=OperationMeta(model="gpt-4o", provider="openai"),
            ),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s1",
                response_id="chatcmpl-abc123",
            ),
        ]

        tree = project_tree(events)
        node = tree.roots[0]
        assert node.operation is not None
        assert node.operation.model == "gpt-4o"
        assert node.response_id == "chatcmpl-abc123"

    def test_error_status(self):
        """project_tree should set FAILED status on ERROR event."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="failing"),
            TraceEvent(type=EventType.ERROR, trace_id="t1", span_id="s1", error="test error"),
        ]

        tree = project_tree(events)
        assert tree.roots[0].status == "FAILED"
        assert tree.roots[0].error == "test error"

    def test_cancel_status(self):
        """project_tree should set CANCELLED status on CANCEL event."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="cancelled"),
            TraceEvent(type=EventType.CANCEL, trace_id="t1", span_id="s1"),
        ]

        tree = project_tree(events)
        assert tree.roots[0].status == "CANCELLED"

    def test_find_by_name(self):
        """TreeView.find_by_name should find nodes by name."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="agent"),
            TraceEvent(
                type=EventType.START,
                span_id="s2",
                name="tool",
                relations={Relation.CONTEXT: ["s1"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s2"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1"),
        ]

        tree = project_tree(events)
        found = tree.find_by_name("tool")
        assert found is not None
        assert found.name == "tool"

    def test_find_by_kind(self):
        """TreeView.find_by_kind should find all nodes of a kind."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="s1", name="agent", kind="agent"
            ),
            TraceEvent(
                type=EventType.START,
                span_id="s2",
                name="tool1",
                kind="tool",
                relations={Relation.CONTEXT: ["s1"]},
            ),
            TraceEvent(
                type=EventType.START,
                span_id="s3",
                name="tool2",
                kind="tool",
                relations={Relation.CONTEXT: ["s1"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s3"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s2"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1"),
        ]

        tree = project_tree(events)
        tools = tree.find_by_kind("tool")
        assert len(tools) == 2

    def test_find_failed(self):
        """TreeView.find_failed should find all failed spans."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="ok"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1"),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s2", name="fail"),
            TraceEvent(type=EventType.ERROR, trace_id="t1", span_id="s2", error="failed"),
        ]

        tree = project_tree(events)
        failed = tree.find_failed()
        assert len(failed) == 1
        assert failed[0].name == "fail"

    def test_find_incomplete(self):
        """TreeView.find_incomplete should find RUNNING/PENDING spans."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="running"),
        ]

        tree = project_tree(events)
        incomplete = tree.find_incomplete()
        assert len(incomplete) == 1
        assert incomplete[0].status in ("RUNNING", "PENDING")

    def test_total_tokens_recursive(self):
        """TreeNode.total_tokens should sum tokens recursively."""
        from saccade.primitives.events import EventType, Relation, TokenMetrics, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="agent", name="agent"),
            TraceEvent(
                type=EventType.START,
                span_id="tool1",
                name="tool1",
                relations={Relation.CONTEXT: ["agent"]},
            ),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="tool1",
                tokens=TokenMetrics(input=100, output=50),
            ),
            TraceEvent(
                type=EventType.START,
                span_id="tool2",
                name="tool2",
                relations={Relation.CONTEXT: ["agent"]},
            ),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="tool2",
                tokens=TokenMetrics(input=200, output=75),
            ),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="agent",
                tokens=TokenMetrics(input=50, output=25),
            ),
        ]

        tree = project_tree(events)
        agent = tree.roots[0]
        assert agent.total_tokens.input == 350
        assert agent.total_tokens.output == 150

    def test_tree_view_total_tokens(self):
        """TreeView.total_tokens should sum all nodes."""
        from saccade.primitives.events import EventType, TokenMetrics, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="s1"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s1",
                tokens=TokenMetrics(input=100, output=50),
            ),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s2", name="s2"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s2",
                tokens=TokenMetrics(input=200, output=75),
            ),
        ]

        tree = project_tree(events)
        assert tree.total_tokens.input == 300
        assert tree.total_tokens.output == 125

    def test_nodes_index(self):
        """TreeView.nodes should provide O(1) lookup by span_id."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="agent", name="agent"),
            TraceEvent(
                type=EventType.START,
                span_id="tool",
                name="tool",
                relations={Relation.CONTEXT: ["agent"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="tool"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="agent"),
        ]

        tree = project_tree(events)
        assert "agent" in tree.nodes
        assert "tool" in tree.nodes
        assert tree.nodes["tool"].name == "tool"


class TestProjectState:
    """Tests for project_state - context reconstruction projection."""

    def test_empty_events(self):
        """project_state should handle empty events."""
        from saccade.primitives.projectors import StateView, project_state

        state = project_state([])
        assert isinstance(state, StateView)
        assert len(state.snapshots) == 0

    def test_state_from_inputs(self):
        """project_state should derive state from START inputs."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_state

        events = [
            TraceEvent(
                type=EventType.START,
                span_id="agent",
                name="agent",
                timestamp=0.0,
                inputs={"query": "search for X", "user_id": "123"},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="agent", timestamp=10.0),
        ]

        state = project_state(events)
        snapshot = state.at_span("agent")
        assert snapshot.state["query"] == "search for X"
        assert snapshot.state["user_id"] == "123"

    def test_state_from_output(self):
        """project_state should derive state from OUTPUT events."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_state

        events = [
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="agent", name="agent", timestamp=0.0
            ),
            TraceEvent(
                type=EventType.OUTPUT,
                trace_id="t1",
                span_id="agent",
                timestamp=10.0,
                output="result",
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="agent", timestamp=20.0),
        ]

        state = project_state(events)
        snapshot = state.at_time(15.0)
        assert snapshot.state["output"] == "result"

    def test_state_from_chunks(self):
        """project_state should accumulate state from CHUNK events."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_state

        events = [
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="agent", name="agent", timestamp=0.0
            ),
            TraceEvent(
                type=EventType.CHUNK, trace_id="t1", span_id="agent", timestamp=10.0, chunk="Hello"
            ),
            TraceEvent(
                type=EventType.CHUNK, trace_id="t1", span_id="agent", timestamp=20.0, chunk=" "
            ),
            TraceEvent(
                type=EventType.CHUNK, trace_id="t1", span_id="agent", timestamp=30.0, chunk="world"
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="agent", timestamp=40.0),
        ]

        state = project_state(events)
        snapshot = state.at_time(35.0)
        assert snapshot.streamed_text == "Hello world"

    def test_at_span(self):
        """StateView.at_span should return state at span's start."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_state

        events = [
            TraceEvent(
                type=EventType.START,
                span_id="agent",
                name="agent",
                timestamp=0.0,
                inputs={"context": "initial"},
            ),
            TraceEvent(
                type=EventType.START,
                span_id="tool",
                name="tool",
                timestamp=15.0,
                relations={Relation.CONTEXT: ["agent"]},
            ),
            TraceEvent(
                type=EventType.OUTPUT,
                span_id="agent",
                timestamp=20.0,
                output="updated",
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="tool", timestamp=25.0),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="agent", timestamp=30.0),
        ]

        state = project_state(events)
        tool_snapshot = state.at_span("tool")
        assert tool_snapshot.state["context"] == "initial"

    def test_multiple_spans_isolated_state(self):
        """project_state should track state per span."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_state

        events = [
            TraceEvent(
                type=EventType.START,
                span_id="agent1",
                name="agent1",
                timestamp=0.0,
                inputs={"x": 1},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="agent1", timestamp=20.0),
            TraceEvent(
                type=EventType.START,
                span_id="agent2",
                name="agent2",
                timestamp=5.0,
                inputs={"y": 2},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="agent2", timestamp=25.0),
        ]

        state = project_state(events)
        agent1_snapshot = state.at_span("agent1")
        agent2_snapshot = state.at_span("agent2")

        assert agent1_snapshot.state == {"x": 1}
        assert agent2_snapshot.state == {"y": 2}


class TestProjectTimeline:
    """Tests for project_timeline - concurrency and latency projection."""

    def test_empty_events(self):
        """project_timeline should handle empty events."""
        from saccade.primitives.projectors import TimelineView, project_timeline

        timeline = project_timeline([])
        assert isinstance(timeline, TimelineView)
        assert len(timeline.spans) == 0

    def test_spans_sorted_by_start_time(self):
        """project_timeline should sort spans by start time."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_timeline

        events = [
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="s2", name="second", timestamp=2000.0
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s2", timestamp=2100.0),
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="s1", name="first", timestamp=1000.0
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1", timestamp=1100.0),
        ]

        timeline = project_timeline(events)
        assert timeline.spans[0].name == "first"
        assert timeline.spans[1].name == "second"

    def test_duration_calculated(self):
        """project_timeline should calculate duration from start to terminal event."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_timeline

        events = [
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="s1", name="s1", timestamp=1000.0
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1", timestamp=1100.0),
        ]

        timeline = project_timeline(events)
        assert timeline.spans[0].duration_ms == 100.0

    def test_depth_from_hierarchy(self):
        """project_timeline should compute depth from context relation."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_timeline

        events = [
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="agent", name="agent", timestamp=0.0
            ),
            TraceEvent(
                type=EventType.START,
                span_id="tool",
                name="tool",
                timestamp=10.0,
                relations={Relation.CONTEXT: ["agent"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="tool", timestamp=50.0),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="agent", timestamp=100.0),
        ]

        timeline = project_timeline(events)
        agent = next(s for s in timeline.spans if s.name == "agent")
        tool = next(s for s in timeline.spans if s.name == "tool")

        assert agent.depth == 0
        assert tool.depth == 1

    def test_total_duration(self):
        """TimelineView.total_duration_ms should span all spans."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_timeline

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="s1", timestamp=0.0),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1", timestamp=100.0),
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="s2", name="s2", timestamp=50.0
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s2", timestamp=200.0),
        ]

        timeline = project_timeline(events)
        assert timeline.start_ms == 0.0
        assert timeline.end_ms == 200.0
        assert timeline.total_duration_ms == 200.0

    def test_running_span_has_no_end(self):
        """project_timeline should handle incomplete spans."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_timeline

        events = [
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="s1", name="s1", timestamp=1000.0
            ),
        ]

        timeline = project_timeline(events)
        assert timeline.spans[0].end_ms is None
        assert timeline.spans[0].status == "RUNNING"

    def test_configurable_hierarchy_relation(self):
        """project_timeline should accept custom hierarchy relation."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_timeline

        events = [
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="parent", name="parent", timestamp=0.0
            ),
            TraceEvent(
                type=EventType.START,
                span_id="child",
                name="child",
                timestamp=10.0,
                relations={"custom_parent": ["parent"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="child", timestamp=50.0),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="parent", timestamp=100.0),
        ]

        timeline = project_timeline(events, hierarchy_relation="custom_parent")
        child = next(s for s in timeline.spans if s.name == "child")
        assert child.depth == 1


class TestProjectGraph:
    """Tests for project_graph - data flow projection."""

    def test_empty_events(self):
        """project_graph should handle empty events."""
        from saccade.primitives.projectors import GraphView, project_graph

        graph = project_graph([])
        assert isinstance(graph, GraphView)
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_includes_dataflow_relations(self):
        """project_graph should include DATAFLOW relations as edges."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_graph

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="search", name="search"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="search"),
            TraceEvent(
                type=EventType.START,
                span_id="summarize",
                name="summarize",
                relations={Relation.DATAFLOW: ["search"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="summarize"),
        ]

        graph = project_graph(events)
        assert len(graph.edges) == 1
        edge = graph.edges[0]
        assert edge.source_id == "summarize"
        assert edge.target_id == "search"
        assert edge.relation_type == Relation.DATAFLOW

    def test_includes_all_dataflow_by_default(self):
        """project_graph should include all DATAFLOW edges."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_graph

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="db", name="db"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="db"),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="search", name="search"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="search"),
            TraceEvent(
                type=EventType.START,
                span_id="generate",
                name="generate",
                relations={Relation.DATAFLOW: ["search", "db"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="generate"),
        ]

        graph = project_graph(events)
        gen_node = graph.nodes["generate"]
        assert len(gen_node.edges_out) == 2

    def test_edges_in_and_out(self):
        """GraphNode should track incoming and outgoing edges."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_graph

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="a", name="a"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="a"),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="b", name="b"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="b"),
            TraceEvent(
                type=EventType.START,
                span_id="c",
                name="c",
                relations={Relation.DATAFLOW: ["a", "b"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="c"),
        ]

        graph = project_graph(events)
        c_node = graph.nodes["c"]
        a_node = graph.nodes["a"]
        b_node = graph.nodes["b"]

        assert len(c_node.edges_out) == 2
        assert len(a_node.edges_in) == 1
        assert len(b_node.edges_in) == 1

    def test_ghost_nodes_for_missing_refs(self):
        """project_graph should create ghost nodes for missing references."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_graph

        events = [
            TraceEvent(
                type=EventType.START,
                span_id="s1",
                name="s1",
                relations={Relation.DATAFLOW: ["nonexistent"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1"),
        ]

        graph = project_graph(events)
        assert len(graph.nodes) == 2

        ghosts = graph.ghosts()
        assert len(ghosts) == 1
        assert ghosts[0].is_ghost
        assert ghosts[0].id == "nonexistent"

    def test_no_ghost_nodes_when_all_exist(self):
        """GraphView.ghosts should be empty when all refs exist."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_graph

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="s1"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1"),
            TraceEvent(
                type=EventType.START,
                span_id="s2",
                name="s2",
                relations={Relation.DATAFLOW: ["s1"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s2"),
        ]

        graph = project_graph(events)
        assert len(graph.ghosts()) == 0

    def test_adjacency_list(self):
        """GraphView.adjacency_list should return adjacency dict."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_graph

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="s1"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1"),
            TraceEvent(
                type=EventType.START,
                span_id="s2",
                name="s2",
                relations={Relation.DATAFLOW: ["s1"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s2"),
        ]

        graph = project_graph(events)
        adj = graph.adjacency_list()
        assert "s2" in adj
        assert "s1" in adj["s2"]

    def test_agent_team_data_flow(self):
        """project_graph should show data flow between agents."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_graph

        events = [
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="orchestrator", name="orchestrator"
            ),
            TraceEvent(
                type=EventType.START,
                span_id="researcher",
                name="researcher",
                relations={Relation.CONTEXT: ["orchestrator"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="researcher"),
            TraceEvent(
                type=EventType.START,
                span_id="analyst",
                name="analyst",
                relations={
                    Relation.CONTEXT: ["orchestrator"],
                    Relation.DATAFLOW: ["researcher"],
                },
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="analyst"),
            TraceEvent(
                type=EventType.START,
                span_id="writer",
                name="writer",
                relations={
                    Relation.CONTEXT: ["orchestrator"],
                    Relation.DATAFLOW: ["analyst"],
                },
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="writer"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="orchestrator"),
        ]

        graph = project_graph(events)

        assert len(graph.edges) == 2

        adj = graph.adjacency_list()
        assert "analyst" in adj
        assert "researcher" in adj["analyst"]
        assert "writer" in adj
        assert "analyst" in adj["writer"]

    def test_hub_and_spoke_data_flow(self):
        """project_graph should show hub-spoke data flow patterns."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_graph

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="agent", name="agent"),
            TraceEvent(
                type=EventType.START,
                span_id="tool1",
                name="tool1",
                relations={Relation.CONTEXT: ["agent"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="tool1"),
            TraceEvent(
                type=EventType.START,
                span_id="tool2",
                name="tool2",
                relations={
                    Relation.CONTEXT: ["agent"],
                    Relation.DATAFLOW: ["tool1"],
                },
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="tool2"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="agent"),
        ]

        graph = project_graph(events)

        assert len(graph.edges) == 1
        adj = graph.adjacency_list()
        assert "tool1" in adj["tool2"]


class TestProjectCost:
    """Tests for project_cost - aggregation projection."""

    def test_empty_events(self):
        """project_cost should handle empty events."""
        from saccade.primitives.projectors import CostView, project_cost

        cost = project_cost([])
        assert isinstance(cost, CostView)
        assert cost.span_count == 0

    def test_total_tokens(self):
        """CostView should aggregate total tokens."""
        from saccade.primitives.events import EventType, TokenMetrics, TraceEvent
        from saccade.primitives.projectors import project_cost

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="s1"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s1",
                tokens=TokenMetrics(input=100, output=50),
            ),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s2", name="s2"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s2",
                tokens=TokenMetrics(input=200, output=75),
            ),
        ]

        cost = project_cost(events)
        assert cost.tokens.input == 300
        assert cost.tokens.output == 125

    def test_total_cost(self):
        """CostView should aggregate total cost."""
        from saccade.primitives.events import CostMetrics, EventType, TraceEvent
        from saccade.primitives.projectors import project_cost

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="s1"),
            TraceEvent(
                type=EventType.SUCCESS,
                trace_id="t1",
                span_id="s1",
                cost=CostMetrics(usd=Decimal("0.01")),
            ),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s2", name="s2"),
            TraceEvent(
                type=EventType.SUCCESS,
                trace_id="t1",
                span_id="s2",
                cost=CostMetrics(usd=Decimal("0.02")),
            ),
        ]

        cost = project_cost(events)
        assert cost.cost.usd == Decimal("0.03")

    def test_by_model(self):
        """CostView should group metrics by model."""
        from saccade.primitives.events import EventType, OperationMeta, TokenMetrics, TraceEvent
        from saccade.primitives.projectors import project_cost

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="s1"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s1",
                tokens=TokenMetrics(input=100, output=50),
                operation=OperationMeta(model="gpt-4o"),
            ),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s2", name="s2"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s2",
                tokens=TokenMetrics(input=200, output=75),
                operation=OperationMeta(model="claude-3"),
            ),
        ]

        cost = project_cost(events)
        assert "gpt-4o" in cost.by_model
        assert "claude-3" in cost.by_model
        assert cost.by_model["gpt-4o"].tokens.input == 100
        assert cost.by_model["claude-3"].tokens.input == 200

    def test_by_kind(self):
        """CostView should group metrics by kind."""
        from saccade.primitives.events import EventType, TokenMetrics, TraceEvent
        from saccade.primitives.projectors import project_cost

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="s1", kind="agent"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s1",
                tokens=TokenMetrics(input=100, output=50),
            ),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s2", name="s2", kind="tool"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s2",
                tokens=TokenMetrics(input=200, output=75),
            ),
        ]

        cost = project_cost(events)
        assert "agent" in cost.by_kind
        assert "tool" in cost.by_kind
        assert cost.by_kind["agent"].tokens.input == 100
        assert cost.by_kind["tool"].tokens.input == 200

    def test_by_name(self):
        """CostView should group metrics by span name."""
        from saccade.primitives.events import EventType, TokenMetrics, TraceEvent
        from saccade.primitives.projectors import project_cost

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="researcher"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s1",
                tokens=TokenMetrics(input=100, output=50),
            ),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s2", name="researcher"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s2",
                tokens=TokenMetrics(input=200, output=75),
            ),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s3", name="writer"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s3",
                tokens=TokenMetrics(input=50, output=25),
            ),
        ]

        cost = project_cost(events)
        assert cost.by_name["researcher"].tokens.input == 300
        assert cost.by_name["researcher"].count == 2
        assert cost.by_name["writer"].tokens.input == 50
        assert cost.by_name["writer"].count == 1

    def test_span_count(self):
        """CostView should count total spans."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_cost

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="s1"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1"),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s2", name="s2"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s2"),
        ]

        cost = project_cost(events)
        assert cost.span_count == 2

    def test_latency_aggregation(self):
        """CostView should aggregate latency metrics."""
        from saccade.primitives.events import EventType, LatencyMetrics, TraceEvent
        from saccade.primitives.projectors import project_cost

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="s1"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s1",
                latency=LatencyMetrics(total_ms=100.0, time_to_first_token_ms=50.0),
            ),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s2", name="s2"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s2",
                latency=LatencyMetrics(total_ms=200.0, time_to_first_token_ms=75.0),
            ),
        ]

        cost = project_cost(events)
        assert cost.latency.total_ms == 300.0


class TestProjectTreeEdgeCases:
    """Edge case tests for project_tree - distributed tracing robustness."""

    def test_orphaned_events_promoted_to_root(self):
        """project_tree should promote events with missing parents to root."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(
                type=EventType.START,
                span_id="orphan",
                name="orphan",
                relations={Relation.CONTEXT: ["ghost_parent"]},
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="orphan"),
        ]

        tree = project_tree(events)

        assert len(tree.roots) == 1
        assert tree.roots[0].name == "orphan"
        assert tree.roots[0].metadata.get("orphan") is True
        assert tree.roots[0].metadata.get("missing_parent") == "ghost_parent"

    def test_orphan_resolved_when_parent_arrives_later(self):
        """project_tree should resolve orphans when parent arrives (eventual consistency)."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(
                type=EventType.START,
                span_id="child",
                name="child",
                timestamp=100.0,
                relations={Relation.CONTEXT: ["parent"]},
            ),
            TraceEvent(
                type=EventType.START,
                span_id="parent",
                name="parent",
                timestamp=50.0,
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="child", timestamp=200.0),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="parent", timestamp=300.0),
        ]

        tree = project_tree(events)

        assert len(tree.roots) == 1
        assert tree.roots[0].name == "parent"
        assert len(tree.roots[0].children) == 1
        assert tree.roots[0].children[0].name == "child"

    def test_duplicate_start_raises_error(self):
        """project_tree should raise error on duplicate START for same span_id."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="first"),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="duplicate"),
        ]

        with pytest.raises(Exception):
            project_tree(events)

    def test_multiple_events_same_span_merged(self):
        """project_tree should merge multiple events for the same span_id."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="streaming"),
            TraceEvent(type=EventType.CHUNK, trace_id="t1", span_id="s1", chunk="Hello"),
            TraceEvent(type=EventType.CHUNK, trace_id="t1", span_id="s1", chunk=" world"),
            TraceEvent(type=EventType.OUTPUT, trace_id="t1", span_id="s1", output="Hello world"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1"),
        ]

        tree = project_tree(events)

        assert len(tree.roots) == 1
        assert tree.roots[0].name == "streaming"
        assert tree.roots[0].streamed_text == "Hello world"
        assert tree.roots[0].output == "Hello world"

    def test_out_of_order_events_sorted_internally(self):
        """project_tree should handle events arriving out of order."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="child", timestamp=200.0),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="parent", timestamp=300.0),
            TraceEvent(
                type=EventType.START,
                span_id="child",
                name="child",
                timestamp=100.0,
                relations={Relation.CONTEXT: ["parent"]},
            ),
            TraceEvent(
                type=EventType.START,
                span_id="parent",
                name="parent",
                timestamp=50.0,
            ),
        ]

        tree = project_tree(events)

        assert len(tree.roots) == 1
        assert tree.roots[0].name == "parent"
        assert tree.roots[0].status == "COMPLETED"
        assert len(tree.roots[0].children) == 1
        assert tree.roots[0].children[0].name == "child"

    def test_concurrent_spans_dont_bleed_chunks(self):
        """project_tree should not mix chunks between concurrent spans."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="a", name="span_a"),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="b", name="span_b"),
            TraceEvent(type=EventType.CHUNK, trace_id="t1", span_id="a", chunk="A1"),
            TraceEvent(type=EventType.CHUNK, trace_id="t1", span_id="b", chunk="B1"),
            TraceEvent(type=EventType.CHUNK, trace_id="t1", span_id="a", chunk="A2"),
            TraceEvent(type=EventType.CHUNK, trace_id="t1", span_id="b", chunk="B2"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="a"),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="b"),
        ]

        tree = project_tree(events)

        span_a = tree.nodes["a"]
        span_b = tree.nodes["b"]

        assert span_a.streamed_text == "A1A2"
        assert span_b.streamed_text == "B1B2"

    def test_circular_dependency_handling(self):
        """project_tree should gracefully handle circular context loops."""
        from saccade.primitives.events import EventType, Relation, TraceEvent
        from saccade.primitives.projectors import project_tree

        events = [
            TraceEvent(
                type=EventType.START,
                span_id="s1",
                name="A",
                relations={Relation.CONTEXT: ["s2"]},
            ),
            TraceEvent(
                type=EventType.START,
                span_id="s2",
                name="B",
                relations={Relation.CONTEXT: ["s1"]},
            ),
        ]

        tree = project_tree(events)
        assert len(tree.nodes) == 2


class TestProjectStateEdgeCases:
    """Edge case tests for project_state - state reconstruction robustness."""

    def test_snapshots_are_independent(self):
        """project_state snapshots should be immutable/independent (deep copy)."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_state

        events = [
            TraceEvent(
                type=EventType.START,
                span_id="s1",
                timestamp=10.0,
                inputs={"data": {"a": 1}},
            ),
            TraceEvent(
                type=EventType.OUTPUT,
                span_id="s1",
                timestamp=20.0,
                output="update",
            ),
        ]

        state_view = project_state(events)

        snap_t15 = state_view.at_time(15.0)
        snap_t25 = state_view.at_time(25.0)

        snap_t25.state["injected"] = True
        snap_t25.state["data"]["b"] = 2

        assert "injected" not in snap_t15.state
        assert "b" not in snap_t15.state["data"]

    def test_custom_reducer_shallow(self):
        """project_state should use custom reducer for state merges."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_state

        def shallow_reducer(current: dict, new: dict) -> dict:
            return {**current, **new}

        events = [
            TraceEvent(
                type=EventType.START,
                span_id="s1",
                timestamp=0.0,
                inputs={"config": {"model": "gpt-4"}},
            ),
            TraceEvent(
                type=EventType.OUTPUT,
                span_id="s1",
                timestamp=10.0,
                output={"config": {"temp": 0.7}},
            ),
        ]

        state = project_state(events, reducer=shallow_reducer)
        snapshot = state.at_time(15.0)

        assert snapshot.state["config"] == {"temp": 0.7}

    def test_custom_reducer_deep_merge(self):
        """project_state should support deep merge via custom reducer."""
        import copy

        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_state

        def deep_merge_reducer(current: dict, new: dict) -> dict:
            result = copy.deepcopy(current)
            for key, value in new.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = {**result[key], **value}
                else:
                    result[key] = value
            return result

        events = [
            TraceEvent(
                type=EventType.START,
                span_id="s1",
                timestamp=0.0,
                inputs={"config": {"model": "gpt-4", "temp": 0.7}},
            ),
            TraceEvent(
                type=EventType.OUTPUT,
                span_id="s1",
                timestamp=10.0,
                output={"config": {"temp": 0.9, "max_tokens": 1000}},
            ),
        ]

        state = project_state(events, reducer=deep_merge_reducer)
        snapshot = state.at_time(15.0)

        assert snapshot.state["config"]["model"] == "gpt-4"
        assert snapshot.state["config"]["temp"] == 0.9
        assert snapshot.state["config"]["max_tokens"] == 1000

    def test_custom_reducer_list_accumulate(self):
        """project_state should support list accumulation via custom reducer."""
        import copy

        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_state

        def list_accumulate_reducer(current: dict, new: dict) -> dict:
            result = copy.deepcopy(current)
            for key, value in new.items():
                if key in result and isinstance(result[key], list) and isinstance(value, list):
                    result[key] = result[key] + value
                else:
                    result[key] = value
            return result

        events = [
            TraceEvent(
                type=EventType.START,
                span_id="s1",
                timestamp=0.0,
                inputs={"messages": [{"role": "user", "content": "hi"}]},
            ),
            TraceEvent(
                type=EventType.OUTPUT,
                span_id="s1",
                timestamp=10.0,
                output={"messages": [{"role": "assistant", "content": "hello"}]},
            ),
        ]

        state = project_state(events, reducer=list_accumulate_reducer)
        snapshot = state.at_time(15.0)

        assert len(snapshot.state["messages"]) == 2
        assert snapshot.state["messages"][0]["role"] == "user"
        assert snapshot.state["messages"][1]["role"] == "assistant"

    def test_default_reducer_is_shallow(self):
        """project_state default reducer should be shallow merge."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_state

        events = [
            TraceEvent(
                type=EventType.START,
                span_id="s1",
                timestamp=0.0,
                inputs={"x": 1, "y": 2},
            ),
            TraceEvent(
                type=EventType.OUTPUT,
                span_id="s1",
                timestamp=10.0,
                output={"y": 3, "z": 4},
            ),
        ]

        state = project_state(events)
        snapshot = state.at_time(15.0)

        assert snapshot.state == {"x": 1, "y": 3, "z": 4}

    def test_non_dict_inputs_handled_safely(self):
        """project_state should not crash on non-dict inputs (defensive coding)."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_state

        events = [
            TraceEvent.model_construct(
                type=EventType.START,
                span_id="s1",
                timestamp=0.0,
                inputs="<Unserializable Object>",
            )
        ]

        state = project_state(events)
        assert state.at_span("s1").state == {}


class TestProjectTimelineEdgeCases:
    """Edge case tests for project_timeline - timing robustness."""

    def test_clock_skew_clamped_to_zero(self):
        """project_timeline should clamp negative duration to 0."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_timeline

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", timestamp=1000.0),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1", timestamp=999.0),
        ]

        timeline = project_timeline(events)
        span = timeline.spans[0]

        assert span.duration_ms >= 0.0

    def test_clock_skew_flagged(self):
        """project_timeline should flag clock skew in span metadata."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_timeline

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", timestamp=1000.0),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1", timestamp=999.0),
        ]

        timeline = project_timeline(events)
        span = timeline.spans[0]

        assert span.clock_skew_detected is True

    def test_normal_duration_not_flagged(self):
        """project_timeline should not flag normal durations."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_timeline

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", timestamp=1000.0),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1", timestamp=1100.0),
        ]

        timeline = project_timeline(events)
        span = timeline.spans[0]

        assert span.clock_skew_detected is False

    def test_out_of_order_events_sorted_internally(self):
        """project_timeline should sort events by timestamp internally."""
        from saccade.primitives.events import EventType, TraceEvent
        from saccade.primitives.projectors import project_timeline

        events = [
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s2", timestamp=2100.0),
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="s1", name="first", timestamp=1000.0
            ),
            TraceEvent(
                type=EventType.START, trace_id="t1", span_id="s2", name="second", timestamp=2000.0
            ),
            TraceEvent(type=EventType.SUCCESS, trace_id="t1", span_id="s1", timestamp=1100.0),
        ]

        timeline = project_timeline(events)

        assert timeline.spans[0].name == "first"
        assert timeline.spans[1].name == "second"


class TestProjectCostEdgeCases:
    """Edge case tests for project_cost - cost aggregation robustness."""

    def test_decimal_precision_no_drift(self):
        """project_cost should use Decimal for precise cost aggregation."""
        from decimal import Decimal

        from saccade.primitives.events import CostMetrics, EventType, TraceEvent
        from saccade.primitives.projectors import project_cost

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="s1"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s1",
                cost=CostMetrics(usd=Decimal("0.1")),
            ),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s2", name="s2"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s2",
                cost=CostMetrics(usd=Decimal("0.2")),
            ),
        ]

        cost = project_cost(events)
        assert cost.cost.usd == Decimal("0.3")

    def test_aggregation_propagates_clock_skew(self):
        """project_cost should propagate clock_skew_detected across spans."""
        from saccade.primitives.events import EventType, LatencyMetrics, TraceEvent
        from saccade.primitives.projectors import project_cost

        events = [
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="s1"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s1",
                latency=LatencyMetrics(total_ms=100.0, has_clock_skew=True),
            ),
            TraceEvent(type=EventType.START, trace_id="t1", span_id="s2", name="s2"),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id="s2",
                latency=LatencyMetrics(total_ms=200.0, has_clock_skew=False),
            ),
        ]

        cost = project_cost(events)
        assert cost.latency.has_clock_skew is True

    def test_duplicate_events_ignored_for_cost(self):
        """project_cost should handle exact duplicate events without double counting."""
        from saccade.primitives.events import CostMetrics, EventType, TraceEvent
        from saccade.primitives.projectors import project_cost

        success_event = TraceEvent(
            type=EventType.SUCCESS,
            span_id="s1",
            cost=CostMetrics(usd=Decimal("0.1")),
        )

        start_event = TraceEvent(type=EventType.START, trace_id="t1", span_id="s1", name="s1")

        cost = project_cost([start_event, success_event, success_event])

        assert cost.cost.usd == Decimal("0.1")
