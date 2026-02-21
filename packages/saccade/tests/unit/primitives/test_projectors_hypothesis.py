import time

from hypothesis import given, settings
from hypothesis import strategies as st

import pytest

from saccade.primitives.events import (
    CostMetrics,
    EventType,
    Relation,
    TokenMetrics,
    TraceEvent,
)
from saccade.primitives.projectors import project_cost, project_tree

pytestmark = pytest.mark.unit


def event_strategy():
    return st.builds(
        TraceEvent,
        type=st.sampled_from(EventType),
        span_id=st.text(
            min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
        ),
        name=st.none() | st.text(min_size=1, max_size=20),
        timestamp=st.floats(min_value=time.time() - 1000, max_value=time.time()),
    )


class TestTreeProjectionInvariants:
    def test_empty_events_produces_empty_tree(self):
        tree = project_tree([])

        assert tree.roots == []
        assert tree.nodes == {}

    @given(span_id=st.text(min_size=1, max_size=10, alphabet="abcdef"))
    def test_single_start_event_creates_node(self, span_id):
        events = [
            TraceEvent(
                type=EventType.START,
                span_id=span_id,
                name="test_span",
                kind="test",
            )
        ]

        tree = project_tree(events)

        assert len(tree.nodes) == 1
        assert span_id in tree.nodes
        assert tree.nodes[span_id].name == "test_span"
        assert tree.nodes[span_id].status == "RUNNING"

    @given(span_id=st.text(min_size=1, max_size=10, alphabet="abcdef"))
    def test_start_then_success_completes_node(self, span_id):
        events = [
            TraceEvent(
                type=EventType.START,
                span_id=span_id,
                name="test_span",
                timestamp=time.time() - 0.1,
            ),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id=span_id,
                timestamp=time.time(),
            ),
        ]

        tree = project_tree(events)

        assert tree.nodes[span_id].status == "COMPLETED"

    @given(
        parent_id=st.text(min_size=1, max_size=5, alphabet="abc"),
        child_id=st.text(min_size=1, max_size=5, alphabet="xyz"),
    )
    @settings(max_examples=30)
    def test_parent_child_relationship(self, parent_id, child_id):
        events = [
            TraceEvent(
                type=EventType.START,
                span_id=parent_id,
                name="parent",
                timestamp=time.time() - 0.2,
            ),
            TraceEvent(
                type=EventType.START,
                span_id=child_id,
                name="child",
                relations={Relation.CONTEXT: [parent_id]},
                timestamp=time.time() - 0.1,
            ),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id=child_id,
                timestamp=time.time() - 0.05,
            ),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id=parent_id,
                timestamp=time.time(),
            ),
        ]

        tree = project_tree(events)

        assert len(tree.roots) == 1
        assert tree.roots[0].id == parent_id
        assert len(tree.roots[0].children) == 1
        assert tree.roots[0].children[0].id == child_id

    @given(
        span_id=st.text(min_size=1, max_size=5, alphabet="abc"),
        chunk_count=st.integers(min_value=0, max_value=10),
    )
    def test_chunks_are_accumulated(self, span_id, chunk_count):
        base_time = time.time()
        events = [
            TraceEvent(
                type=EventType.START,
                span_id=span_id,
                name="streaming",
                timestamp=base_time,
            )
        ]

        for i in range(chunk_count):
            events.append(
                TraceEvent(
                    type=EventType.CHUNK,
                    span_id=span_id,
                    chunk=f"chunk_{i}",
                    timestamp=base_time + 0.01 * (i + 1),
                )
            )

        events.append(
            TraceEvent(
                type=EventType.SUCCESS,
                span_id=span_id,
                timestamp=base_time + 0.01 * (chunk_count + 1),
            )
        )

        tree = project_tree(events)

        assert len(tree.nodes[span_id].chunks) == chunk_count
        expected_text = "".join(f"chunk_{i}" for i in range(chunk_count))
        assert tree.nodes[span_id].streamed_text == expected_text


class TestCostProjectionInvariants:
    @given(events=st.lists(event_strategy(), max_size=20))
    def test_cost_view_has_total_tokens(self, events):
        cost_view = project_cost(events)

        assert hasattr(cost_view, "tokens")
        assert hasattr(cost_view, "cost")

    @given(
        span_id=st.text(min_size=1, max_size=5, alphabet="abc"),
        input_tokens=st.integers(min_value=0, max_value=10000),
        output_tokens=st.integers(min_value=0, max_value=10000),
    )
    def test_single_span_metrics_aggregated(self, span_id, input_tokens, output_tokens):
        events = [
            TraceEvent(
                type=EventType.START,
                span_id=span_id,
                name="test",
                timestamp=time.time() - 0.1,
            ),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id=span_id,
                tokens=TokenMetrics(input=input_tokens, output=output_tokens),
                timestamp=time.time(),
            ),
        ]

        cost_view = project_cost(events)

        assert cost_view.tokens.input == input_tokens
        assert cost_view.tokens.output == output_tokens

    @given(
        span_id1=st.just("span1"),
        span_id2=st.just("span2"),
        input1=st.integers(min_value=0, max_value=1000),
        output1=st.integers(min_value=0, max_value=1000),
        input2=st.integers(min_value=0, max_value=1000),
        output2=st.integers(min_value=0, max_value=1000),
    )
    def test_multiple_spans_metrics_summed(
        self, span_id1, span_id2, input1, output1, input2, output2
    ):
        base_time = time.time()
        events = [
            TraceEvent(
                type=EventType.START,
                span_id=span_id1,
                name="span1",
                timestamp=base_time - 0.2,
            ),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id=span_id1,
                tokens=TokenMetrics(input=input1, output=output1),
                timestamp=base_time - 0.1,
            ),
            TraceEvent(
                type=EventType.START,
                span_id=span_id2,
                name="span2",
                timestamp=base_time - 0.05,
            ),
            TraceEvent(
                type=EventType.SUCCESS,
                span_id=span_id2,
                tokens=TokenMetrics(input=input2, output=output2),
                timestamp=base_time,
            ),
        ]

        cost_view = project_cost(events)

        assert cost_view.tokens.input == input1 + input2
        assert cost_view.tokens.output == output1 + output2


class TestTreeStructureInvariants:
    @given(
        num_roots=st.integers(min_value=1, max_value=5),
        children_per_root=st.integers(min_value=0, max_value=3),
    )
    @settings(max_examples=30)
    def test_roots_have_no_parents_in_tree(self, num_roots, children_per_root):
        base_time = time.time()
        events = []

        for i in range(num_roots):
            root_id = f"root_{i}"
            events.append(
                TraceEvent(
                    type=EventType.START,
                    span_id=root_id,
                    name=f"root_{i}",
                    timestamp=base_time + i * 0.1,
                )
            )

            for j in range(children_per_root):
                child_id = f"root_{i}_child_{j}"
                events.append(
                    TraceEvent(
                        type=EventType.START,
                        span_id=child_id,
                        name=child_id,
                        relations={Relation.CONTEXT: [root_id]},
                        timestamp=base_time + i * 0.1 + 0.01 * (j + 1),
                    )
                )

        tree = project_tree(events)

        root_ids_in_tree = {r.id for r in tree.roots}
        all_node_ids = set(tree.nodes.keys())

        child_ids = set()
        for root in tree.roots:
            for child in root.children:
                child_ids.add(child.id)

        assert root_ids_in_tree == (all_node_ids - child_ids)

    @given(
        depth=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=20)
    def test_nested_children_chain(self, depth):
        base_time = time.time()
        events = []

        prev_id = None
        for i in range(depth):
            span_id = f"level_{i}"
            relations = {Relation.CONTEXT: [prev_id]} if prev_id else None

            events.append(
                TraceEvent(
                    type=EventType.START,
                    span_id=span_id,
                    name=span_id,
                    relations=relations,
                    timestamp=base_time + i * 0.1,
                )
            )
            prev_id = span_id

        tree = project_tree(events)

        current = tree.roots[0]
        for i in range(depth - 1):
            assert len(current.children) == 1
            current = current.children[0]
