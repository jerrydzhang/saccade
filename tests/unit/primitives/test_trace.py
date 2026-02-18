"""Tests for Trace - the high-level entry point for tracing."""

import pytest
import asyncio

pytestmark = pytest.mark.unit


class TestTrace:
    """Tests for Trace sync context manager."""

    def test_trace_lifecycle(self):
        """Trace should enter and exit cleanly."""
        from cadence.primitives.trace import Trace

        with Trace() as trace:
            assert trace.bus is not None

    def test_trace_sets_bus_context(self):
        """Trace should set the bus in context variable."""
        from cadence.primitives.trace import Trace, _current_bus

        assert _current_bus.get() is None

        with Trace() as trace:
            assert _current_bus.get() is trace.bus

        assert _current_bus.get() is None

    def test_events_property(self):
        """trace.events should return raw events list."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span

        with Trace() as trace:
            with Span("agent") as agent:
                agent.set_output("done")

            events = trace.events
            assert len(events) >= 2


class TestTraceWithSpans:
    """Tests for Trace with nested spans."""

    def test_simple_nested_structure(self):
        """Trace should capture nested span structure."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.projectors import project_tree

        with Trace() as trace:
            with Span("agent") as agent:
                with Span("tool") as tool:
                    tool.set_output("result")
                agent.set_output("done")

            tree = project_tree(trace.events)
            assert len(tree.roots) == 1
            assert tree.roots[0].name == "agent"
            assert len(tree.roots[0].children) == 1
            assert tree.roots[0].children[0].name == "tool"

    def test_sequential_spans_are_siblings(self):
        """Sequential spans at same level should be siblings."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.projectors import project_tree

        with Trace() as trace:
            with Span("step1"):
                pass
            with Span("step2"):
                pass
            with Span("step3"):
                pass

            tree = project_tree(trace.events)
            assert len(tree.roots) == 3

    def test_span_captures_events(self):
        """Span should emit START, OUTPUT, SUCCESS events."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.events import EventType

        with Trace() as trace:
            with Span("test") as span:
                span.set_output("result")

            events = trace.events
            event_types = [e.type for e in events if e.span_id == span.id]
            assert EventType.START in event_types
            assert EventType.OUTPUT in event_types
            assert EventType.SUCCESS in event_types

    def test_error_captured(self):
        """Trace should capture span errors."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.projectors import project_tree

        with Trace() as trace:
            with pytest.raises(ValueError):
                with Span("failing"):
                    raise ValueError("test error")

            tree = project_tree(trace.events)
            failed = tree.find_failed()
            assert len(failed) == 1
            assert failed[0].name == "failing"
            assert "test error" in failed[0].error


class TestTraceParallelExecution:
    """Tests for Trace with parallel execution (async span bodies)."""

    @pytest.mark.asyncio
    async def test_parallel_children(self):
        """Parallel spans should all be children of parent."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.projectors import project_tree

        # Trace is sync context manager, but body can be async
        with Trace() as trace:
            with Span("orchestrator") as orch:

                async def worker(name: str):
                    with Span(name, kind="worker") as w:
                        await asyncio.sleep(0.005)
                        w.set_output(f"{name} done")

                async with asyncio.TaskGroup() as tg:
                    tg.create_task(worker("w1"))
                    tg.create_task(worker("w2"))
                    tg.create_task(worker("w3"))

                orch.set_output("all done")

            tree = project_tree(trace.events)
            assert len(tree.roots) == 1
            assert tree.roots[0].name == "orchestrator"
            assert len(tree.roots[0].children) == 3


class TestTraceProjections:
    """Tests for projecting trace events."""

    def test_project_graph(self):
        """project_graph should return GraphView."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.events import Relation
        from cadence.primitives.projectors import project_graph, GraphView

        with Trace() as trace:
            with Span("step1") as s1:
                s1_id = s1.id
            with Span("step2", relations={Relation.DATAFLOW: [s1_id]}):
                pass

            graph = project_graph(trace.events)
            assert isinstance(graph, GraphView)
            assert len(graph.nodes) == 2

    def test_project_cost(self):
        """project_cost should return CostView."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.events import TokenMetrics
        from cadence.primitives.projectors import project_cost, CostView

        with Trace() as trace:
            with Span("llm") as s:
                s.set_metrics(tokens=TokenMetrics(input=100, output=50))

            cost = project_cost(trace.events)
            assert isinstance(cost, CostView)
            assert cost.tokens.input == 100


class TestTraceEdgeCases:
    """Edge case tests for Trace - robustness."""

    def test_nested_traces_isolation(self):
        """Nested traces should not corrupt each other's context."""
        from cadence.primitives.trace import Trace, _current_bus

        with Trace() as t1:
            assert _current_bus.get() == t1.bus

            with Trace() as t2:
                assert _current_bus.get() == t2.bus
                assert t1.bus != t2.bus

            assert _current_bus.get() == t1.bus

    def test_dangling_span_warns(self):
        """Trace should warn when span is still running on exit."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with Trace():
                span = Span("dangling")
                span.__enter__()

            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) > 0
            assert (
                "dangling" in str(runtime_warnings[0].message).lower()
                or "running" in str(runtime_warnings[0].message).lower()
            )


class TestTraceSubscribe:
    """Tests for Trace real-time subscription."""

    def test_subscribe_receives_events(self):
        """Subscribe should receive events in real-time."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span

        received = []

        with Trace() as trace:
            trace.subscribe(lambda e: received.append(e))

            with Span("test") as span:
                span.set_output("result")

        # Events should be received immediately (sync emit)
        assert len(received) >= 2

    def test_subscribe_with_multiple_subscribers(self):
        """Multiple subscribers should all receive events."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span

        received1 = []
        received2 = []

        with Trace() as trace:
            trace.subscribe(lambda e: received1.append(e))
            trace.subscribe(lambda e: received2.append(e))

            with Span("test"):
                pass

        assert len(received1) >= 1
        assert len(received2) >= 1

    def test_subscribe_error_isolation(self):
        """Subscriber errors should not crash the trace."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span

        received = []

        def bad_callback(e):
            raise RuntimeError("Subscriber error")

        def good_callback(e):
            received.append(e)

        with Trace() as trace:
            trace.subscribe(bad_callback)
            trace.subscribe(good_callback)

            with Span("test"):
                pass

        # Good callback should still receive events
        assert len(received) >= 1
