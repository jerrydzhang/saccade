"""Integration tests for Trace → Span → Bus → Projections pipeline."""

import asyncio
from decimal import Decimal

import pytest

pytestmark = pytest.mark.integration


class TestSpanBusIntegration:
    """Tests for Span emitting events to TraceBus."""

    def test_span_emits_to_bus(self):
        """Span should emit START, OUTPUT, SUCCESS events to the bus."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.events import EventType

        received = []

        with Trace() as trace:
            trace.subscribe(lambda e: received.append(e))

            with Span("test") as span:
                span.set_output("result")

        event_types = [e.type for e in received if e.span_id == span.id]
        assert EventType.START in event_types
        assert EventType.OUTPUT in event_types
        assert EventType.SUCCESS in event_types

    def test_span_streaming_emits_chunks(self):
        """Span.stream() should emit CHUNK events to the bus."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.events import EventType

        received = []

        with Trace() as trace:
            trace.subscribe(lambda e: received.append(e))

            with Span("streaming") as span:
                span.stream("Hello")
                span.stream(" ")
                span.stream("world")

        chunks = [e.chunk for e in received if e.type == EventType.CHUNK]
        assert chunks == ["Hello", " ", "world"]

    def test_span_metrics_in_end_event(self):
        """Span metrics should be included in SUCCESS event."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.events import EventType, TokenMetrics, CostMetrics

        received = []

        with Trace() as trace:
            trace.subscribe(lambda e: received.append(e))

            with Span("llm") as span:
                span.set_metrics(tokens=TokenMetrics(input=100, output=50))
                span.set_metrics(cost=CostMetrics(usd=Decimal("0.01")))

        success_event = next(e for e in received if e.type == EventType.SUCCESS)
        assert success_event.tokens.input == 100
        assert success_event.tokens.output == 50
        assert success_event.cost.usd == Decimal("0.01")


class TestTracePipelineIntegration:
    """Tests for full Trace → Span → Bus → Projection pipeline."""

    def test_trace_to_tree_projection(self):
        """Trace with spans should produce correct tree projection."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.projectors import project_tree

        with Trace() as trace:
            with Span("agent") as agent:
                with Span("tool1") as tool1:
                    tool1.set_output("result1")
                with Span("tool2") as tool2:
                    tool2.set_output("result2")
                agent.set_output("done")

            tree = project_tree(trace.events)
            assert len(tree.roots) == 1
            assert tree.roots[0].name == "agent"
            assert len(tree.roots[0].children) == 2
            assert tree.roots[0].children[0].name == "tool1"
            assert tree.roots[0].children[1].name == "tool2"

    def test_trace_to_graph_projection_with_dataflow(self):
        """Trace with dataflow relations should produce correct graph."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.projectors import project_graph

        with Trace() as trace:
            with Span("search") as search:
                search_id = search.id

            with Span("generate", relations={"dataflow": [search_id]}) as gen:
                gen.set_output("answer")

            graph = project_graph(trace.events)
            assert len(graph.edges) == 1

    def test_trace_aggregates_metrics(self):
        """Trace should aggregate metrics from all spans."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.events import TokenMetrics
        from cadence.primitives.projectors import project_tree

        with Trace() as trace:
            with Span("llm1") as s1:
                s1.set_metrics(tokens=TokenMetrics(input=100, output=50))

            with Span("llm2") as s2:
                s2.set_metrics(tokens=TokenMetrics(input=200, output=75))

            tree = project_tree(trace.events)
            assert tree.total_tokens.input == 300
            assert tree.total_tokens.output == 125

    def test_trace_error_propagates_to_projection(self):
        """Span errors should appear in projections."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.projectors import project_tree

        with Trace() as trace:
            with pytest.raises(ValueError):
                with Span("failing"):
                    raise ValueError("something went wrong")

            tree = project_tree(trace.events)
            failed = tree.find_failed()
            assert len(failed) == 1
            assert failed[0].error == "something went wrong"


class TestAsyncContextPropagation:
    """Tests for context propagation across async boundaries."""

    @pytest.mark.asyncio
    async def test_context_propagates_to_async_tasks(self):
        """Parent span context should be visible in async tasks."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.projectors import project_tree

        with Trace() as trace:
            with Span("orchestrator") as orch:
                orch_id = orch.id

                async def worker(name: str):
                    with Span(name) as w:
                        # Verify context propagated
                        assert "context" in w.relations
                        assert orch_id in w.relations["context"]
                        w.set_output(f"{name} done")

                async with asyncio.TaskGroup() as tg:
                    tg.create_task(worker("w1"))
                    tg.create_task(worker("w2"))

                orch.set_output("all done")

            tree = project_tree(trace.events)
            assert len(tree.roots) == 1
            assert len(tree.roots[0].children) == 2

    def test_bus_subscribers_receive_all_events(self):
        """All subscribers should receive all events."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span

        received1 = []
        received2 = []

        with Trace() as trace:
            trace.subscribe(lambda e: received1.append(e.span_id))
            trace.subscribe(lambda e: received2.append(e.type.value))

            with Span("test") as s:
                s.set_output("result")

            assert len(received1) >= 2
            assert len(received2) >= 2
            assert all(sid == s.id for sid in received1)


class TestErrorHandlingIntegration:
    """Tests for error handling across components."""

    def test_subscriber_error_doesnt_break_pipeline(self):
        """Bus subscriber errors should not break the event pipeline."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span

        good_received = []

        with Trace() as trace:

            def bad_callback(event):
                raise RuntimeError("Subscriber error")

            def good_callback(event):
                good_received.append(event)

            trace.subscribe(bad_callback)
            trace.subscribe(good_callback)

            with Span("test"):
                pass

        assert len(good_received) >= 2

    def test_nested_span_error_captured_correctly(self):
        """Errors in nested spans should be captured at correct level."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.projectors import project_tree

        with Trace() as trace:
            with Span("parent"):
                with pytest.raises(ValueError):
                    with Span("child"):
                        raise ValueError("child error")

            tree = project_tree(trace.events)
            assert tree.roots[0].status == "COMPLETED"
            child = tree.roots[0].children[0]
            assert child.status == "FAILED"
            assert child.error == "child error"


class TestStreamingIntegration:
    """Tests for streaming functionality across components."""

    def test_streaming_accumulates_in_projection(self):
        """Streamed chunks should accumulate in tree projection."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.projectors import project_tree

        with Trace() as trace:
            with Span("streaming") as s:
                s.stream("Hello")
                s.stream(" ")
                s.stream("world")
                s.set_output("Hello world")

            tree = project_tree(trace.events)
            node = tree.roots[0]
            assert node.chunks == ["Hello", " ", "world"]
            assert node.streamed_text == "Hello world"

    @pytest.mark.asyncio
    async def test_streaming_with_metrics(self):
        """Streaming spans should capture TTFT in metrics."""
        from cadence.primitives.trace import Trace
        from cadence.primitives.span import Span
        from cadence.primitives.projectors import project_tree

        with Trace() as trace:
            with Span("streaming") as s:
                await asyncio.sleep(0.005)
                s.stream("first")
                s.stream("second")

            tree = project_tree(trace.events)
            node = tree.roots[0]
            assert node.latency.time_to_first_token_ms is not None
            assert node.latency.time_to_first_token_ms >= 5
