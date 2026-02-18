"""E2E tests for developer experience - realistic user scenarios.

These tests validate that users can accomplish real tasks with minimal
boilerplate using only the public API. They test ergonomics, not internals.
"""

import asyncio
from decimal import Decimal

import pytest

pytestmark = pytest.mark.e2e


class TestMultiTraceScenarios:
    """Tests for multiple concurrent traces remaining isolated."""

    @pytest.mark.asyncio
    async def test_independent_traces_dont_interfere(self):
        """Multiple traces running concurrently should remain isolated."""
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        results = {}

        async def run_agent(agent_id: str):
            with Trace() as trace:
                with Span("work") as s:
                    s.set_output(f"result-{agent_id}")
                results[agent_id] = trace

        async with asyncio.TaskGroup() as tg:
            tg.create_task(run_agent("a"))
            tg.create_task(run_agent("b"))
            tg.create_task(run_agent("c"))

        # Each trace should have its own events with unique trace_ids
        trace_ids = set()
        for agent_id, trace in results.items():
            events = trace.events
            assert all(e.trace_id == trace.trace_id for e in events)
            trace_ids.add(trace.trace_id)
            outputs = [e.output for e in events if e.output is not None]
            assert f"result-{agent_id}" in outputs

        assert len(trace_ids) == 3

    def test_sequential_traces_produce_independent_results(self):
        """Sequential traces should not carry over state from previous runs."""
        from saccade.primitives.projectors import project_tree
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        traces = []

        for i in range(3):
            with Trace() as trace:
                with Span("job") as s:
                    s.set_output({"job_id": i, "status": "done"})
                traces.append(trace)

        trace_ids = [t.trace_id for t in traces]
        assert len(set(trace_ids)) == 3

        for i, trace in enumerate(traces):
            tree = project_tree(trace.events)
            assert len(tree.nodes) == 1
            outputs = [e.output for e in trace.events if e.output is not None]
            assert outputs[0]["job_id"] == i


class TestTraceIDCorrelation:
    """Tests for user-provided trace IDs enabling external correlation."""

    def test_user_provided_trace_id_flows_through(self):
        """User-provided trace_id appears in all events for correlation."""
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        external_request_id = "req-abc123-session-xyz"

        with Trace(trace_id=external_request_id) as trace:
            with Span("step1"):
                pass
            with Span("step2"):
                pass
            with Span("step3"):
                pass

        assert len(trace.events) >= 6
        for event in trace.events:
            assert event.trace_id == external_request_id

    def test_trace_id_matches_across_spans_and_projections(self):
        """Trace ID is consistent across spans and visible in projections."""
        from saccade.primitives.projectors import project_cost, project_tree
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        trace_id = "unified-trace-123"

        with Trace(trace_id=trace_id) as trace:
            with Span("a"):
                pass
            with Span("b"):
                pass

        tree = project_tree(trace.events)
        cost = project_cost(trace.events)

        assert len(tree.nodes) == 2
        assert cost.span_count == 2

        for span_id in tree.nodes:
            matching_events = [e for e in trace.events if e.span_id == span_id]
            assert all(e.trace_id == trace_id for e in matching_events)


class TestErrorRecoveryPatterns:
    """Tests for error scenarios that still produce useful traces."""

    def test_partial_results_captured_before_failure(self):
        """Span captures output even when exception occurs after set_output."""
        from saccade.primitives.projectors import project_tree
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        with Trace() as trace:
            with pytest.raises(RuntimeError):  # noqa: PT012
                with Span("flaky_operation") as s:
                    s.set_output({"partial": "data", "progress": 0.5})
                    raise RuntimeError("failed after progress")

        tree = project_tree(trace.events)
        failed_spans = tree.find_failed()
        assert len(failed_spans) == 1

        failed = failed_spans[0]
        assert failed.output == {"partial": "data", "progress": 0.5}
        assert failed.error == "failed after progress"

    def test_nested_error_captures_context_at_each_level(self):
        """Error in nested span captures state at each level of hierarchy."""
        from saccade.primitives.projectors import project_tree
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        with Trace() as trace:
            with pytest.raises(ValueError):  # noqa: PT012
                with Span("agent") as agent:
                    agent.set_output({"current_step": "processing"})

                    with Span("step1") as s1:
                        s1.set_output({"files": ["a.txt", "b.txt"]})

                    with Span("step2") as s2:
                        s2.set_output({"attempted": "transformation"})
                        raise ValueError("Invalid format detected")

        tree = project_tree(trace.events)
        agent_node = tree.roots[0]

        assert agent_node.output == {"current_step": "processing"}
        assert agent_node.status == "FAILED"

        step1 = tree.find_by_name("step1")
        assert step1.status == "COMPLETED"
        assert step1.output == {"files": ["a.txt", "b.txt"]}

        step2 = tree.find_by_name("step2")
        assert step2.status == "FAILED"
        assert step2.output == {"attempted": "transformation"}
        assert "Invalid format" in step2.error

    def test_error_doesnt_corrupt_sibling_spans(self):
        """Error in one span doesn't affect sibling spans' results."""
        from saccade.primitives.projectors import project_tree
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        with Trace() as trace:
            with Span("orchestrator") as orch:

                def tool(name: str, *, should_fail: bool = False):
                    with Span(name) as s:
                        s.set_output(f"{name} result")
                        if should_fail:
                            err_msg = f"{name} failed"
                            raise RuntimeError(err_msg)

                tool("tool_a", should_fail=False)

                with pytest.raises(RuntimeError):
                    tool("tool_b", should_fail=True)

                orch.set_output("completed with errors")

        tree = project_tree(trace.events)
        tool_a = tree.find_by_name("tool_a")
        tool_b = tree.find_by_name("tool_b")

        assert tool_a.status == "COMPLETED"
        assert tool_a.output == "tool_a result"

        assert tool_b.status == "FAILED"
        assert tool_b.output == "tool_b result"
        assert "tool_b failed" in tool_b.error


class TestCostTrackingPatterns:
    """Tests for realistic cost attribution scenarios."""

    def test_cost_across_multiple_model_calls(self):
        """Multiple LLM calls aggregate costs with Decimal precision."""
        from saccade.primitives.events import CostMetrics, TokenMetrics
        from saccade.primitives.projectors import project_cost
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        with Trace() as trace:
            with Span("call1") as s1:
                s1.set_metrics(
                    tokens=TokenMetrics(input=100, output=50),
                    cost=CostMetrics(usd=Decimal("0.001")),
                )

            with Span("call2") as s2:
                s2.set_metrics(
                    tokens=TokenMetrics(input=200, output=100),
                    cost=CostMetrics(usd=Decimal("0.003")),
                )

            with Span("call3") as s3:
                s3.set_metrics(
                    tokens=TokenMetrics(input=50, output=25),
                    cost=CostMetrics(usd=Decimal("0.0005")),
                )

        cost = project_cost(trace.events)

        assert cost.cost.usd == Decimal("0.0045")
        assert cost.tokens.input == 350
        assert cost.tokens.output == 175
        assert cost.span_count == 3

    def test_cost_by_model_attribution(self):
        """Cost correctly attributed by model for multi-model traces."""
        from saccade.primitives.events import CostMetrics, OperationMeta, TokenMetrics
        from saccade.primitives.projectors import project_cost
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        with Trace() as trace:
            with Span("complex_reasoning") as s:
                s.set_metrics(
                    tokens=TokenMetrics(input=1000, output=500),
                    cost=CostMetrics(usd=Decimal("0.15")),
                )
                s.set_meta(OperationMeta(model="gpt-4o"))

            with Span("simple_task_1") as s:
                s.set_metrics(
                    tokens=TokenMetrics(input=200, output=100),
                    cost=CostMetrics(usd=Decimal("0.001")),
                )
                s.set_meta(OperationMeta(model="gpt-3.5-turbo"))

            with Span("simple_task_2") as s:
                s.set_metrics(
                    tokens=TokenMetrics(input=300, output=150),
                    cost=CostMetrics(usd=Decimal("0.0015")),
                )
                s.set_meta(OperationMeta(model="gpt-3.5-turbo"))

        cost = project_cost(trace.events)

        assert "gpt-4o" in cost.by_model
        assert "gpt-3.5-turbo" in cost.by_model

        assert cost.by_model["gpt-4o"].cost.usd == Decimal("0.15")
        assert cost.by_model["gpt-3.5-turbo"].cost.usd == Decimal("0.0025")

    def test_cost_by_kind_attribution(self):
        """Cost correctly attributed by span kind for analysis."""
        from saccade.primitives.events import CostMetrics, SpanKind, TokenMetrics
        from saccade.primitives.projectors import project_cost
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        with Trace() as trace:
            with Span("planning", kind=SpanKind.AGENT) as s:
                s.set_metrics(
                    tokens=TokenMetrics(input=500, output=200),
                    cost=CostMetrics(usd=Decimal("0.07")),
                )

            with Span("search", kind=SpanKind.TOOL) as s:
                pass

            with Span("generation", kind=SpanKind.LLM) as s:
                s.set_metrics(
                    tokens=TokenMetrics(input=800, output=400),
                    cost=CostMetrics(usd=Decimal("0.12")),
                )

            with Span("embedding", kind=SpanKind.EMBEDDING) as s:
                s.set_metrics(
                    tokens=TokenMetrics(input=1000, output=0),
                    cost=CostMetrics(usd=Decimal("0.001")),
                )

        cost = project_cost(trace.events)

        assert cost.by_kind[SpanKind.AGENT].cost.usd == Decimal("0.07")
        assert cost.by_kind[SpanKind.LLM].cost.usd == Decimal("0.12")
        assert cost.by_kind[SpanKind.EMBEDDING].cost.usd == Decimal("0.001")
        assert SpanKind.TOOL in cost.by_kind
        assert cost.by_kind[SpanKind.TOOL].cost.usd == Decimal("0.0")


class TestMinimalBoilerplate:
    """Tests that validate the happy path requires minimal code."""

    def test_complete_agent_trace_in_five_lines(self):
        """A complete traced agent workflow should be expressible concisely."""
        from saccade.primitives.events import SpanKind
        from saccade.primitives.projectors import project_tree
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        with Trace() as trace:
            with Span("agent", kind=SpanKind.AGENT) as agent:
                with Span("tool", kind=SpanKind.TOOL) as tool:
                    tool.set_output("result")
                agent.set_output("done")

        tree = project_tree(trace.events)

        assert len(tree.nodes) == 2
        assert len(trace.events) >= 4

        agent_node = tree.roots[0]
        assert agent_node.name == "agent"
        assert len(agent_node.children) == 1
        assert agent_node.children[0].name == "tool"

    def test_projection_access_is_explicit(self):
        """Projections accessed through direct function calls."""
        from saccade.primitives.events import CostMetrics, TokenMetrics
        from saccade.primitives.projectors import project_cost, project_tree
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        with Trace() as trace:
            with Span("work") as s:
                s.set_metrics(
                    tokens=TokenMetrics(input=100, output=50),
                    cost=CostMetrics(usd=Decimal("0.01")),
                )

        tree = project_tree(trace.events)
        assert len(tree.nodes) == 1

        cost = project_cost(trace.events)
        assert cost.cost.usd == Decimal("0.01")

    def test_streaming_with_automatic_accumulation(self):
        """Streamed chunks automatically accumulate in projections."""
        from saccade.primitives.projectors import project_tree
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        with Trace() as trace:
            with Span("streaming") as s:
                chunks = ["The ", "quick ", "brown ", "fox"]
                for chunk in chunks:
                    s.stream(chunk)
                s.set_output("The quick brown fox")

        tree = project_tree(trace.events)
        node = tree.roots[0]

        assert node.chunks == chunks
        assert node.streamed_text == "The quick brown fox"
        assert node.output == "The quick brown fox"


class TestRealtimeSubscription:
    """Tests for real-time event subscription."""

    def test_subscribe_receives_events_immediately(self):
        """Subscribers receive events immediately during emit."""
        from saccade.primitives.events import EventType
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        received = []

        with Trace() as trace:
            trace.subscribe(lambda e: received.append(e))

            with Span("work") as s:
                s.set_output("done")

        assert len(received) >= 2
        event_types = [e.type for e in received]
        assert EventType.START in event_types
        assert EventType.SUCCESS in event_types

    def test_subscriber_filtering(self):
        """Subscribers can filter events they care about."""
        from saccade.primitives.events import EventType
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        errors = []

        def on_error(event):
            if event.type == EventType.ERROR:
                errors.append(event)

        with Trace() as trace:
            trace.subscribe(on_error)

            with Span("good"):
                pass

            with pytest.raises(ValueError):
                with Span("bad"):
                    raise ValueError("oops")

        assert len(errors) == 1
        assert errors[0].error == "oops"


class TestRealisticAgentWorkflows:
    """Tests for complete, realistic agent usage patterns."""

    def test_research_agent_workflow(self):
        """Complete research agent workflow with all primitives used."""
        from saccade.primitives.events import (
            CostMetrics,
            OperationMeta,
            SpanKind,
            TokenMetrics,
        )
        from saccade.primitives.projectors import project_cost, project_graph, project_tree
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        with Trace(trace_id="research-001") as trace:
            with Span("research_agent", kind=SpanKind.AGENT) as agent:
                agent.set_output({"task": "Research AI safety"})

                with Span("planning", kind=SpanKind.LLM) as plan:
                    plan.set_metrics(
                        tokens=TokenMetrics(input=200, output=100),
                        cost=CostMetrics(usd=Decimal("0.03")),
                    )
                    plan.set_meta(OperationMeta(model="gpt-4o"))
                    plan.set_output(["search web", "summarize", "write report"])

                search_ids = []
                for topic in ["AI alignment", "AI safety research"]:
                    with Span("search", kind=SpanKind.TOOL) as s:
                        s.set_output(f"Results for {topic}")
                        search_ids.append(s.id)

                with Span(
                    "synthesize",
                    kind=SpanKind.LLM,
                    relations={"dataflow": search_ids},
                ) as synth:
                    synth.set_metrics(
                        tokens=TokenMetrics(input=1500, output=500),
                        cost=CostMetrics(usd=Decimal("0.10")),
                    )
                    synth.set_meta(OperationMeta(model="gpt-4o"))

                with Span("generate_report", kind=SpanKind.LLM) as report:
                    report.set_metrics(
                        tokens=TokenMetrics(input=600, output=800),
                        cost=CostMetrics(usd=Decimal("0.08")),
                    )
                    report.set_meta(OperationMeta(model="gpt-4o"))
                    report.set_output("# AI Safety Report\n\n...")

        tree = project_tree(trace.events)
        assert len(tree.nodes) == 6

        agent_node = tree.roots[0]
        assert len(agent_node.children) == 5

        cost = project_cost(trace.events)
        assert cost.cost.usd == Decimal("0.21")
        assert cost.by_kind[SpanKind.LLM].cost.usd == Decimal("0.21")
        assert cost.by_kind[SpanKind.TOOL].count == 2

        graph = project_graph(trace.events)
        dataflow_edges = [e for e in graph.edges if e.relation_type == "dataflow"]
        assert len(dataflow_edges) == 2

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self):
        """Agent executes tools concurrently with correct trace structure."""
        from saccade.primitives.events import SpanKind
        from saccade.primitives.projectors import project_tree
        from saccade.primitives.span import Span
        from saccade.primitives.trace import Trace

        with Trace() as trace:
            with Span("orchestrator", kind=SpanKind.AGENT) as orch:

                async def execute_tool(name: str, duration: float):
                    with Span(name, kind=SpanKind.TOOL) as s:
                        await asyncio.sleep(duration)
                        s.set_output(f"{name} completed")

                async with asyncio.TaskGroup() as tg:
                    tg.create_task(execute_tool("database_query", 0.01))
                    tg.create_task(execute_tool("api_call", 0.01))
                    tg.create_task(execute_tool("file_read", 0.01))

                orch.set_output("all tools completed")

        tree = project_tree(trace.events)

        orch_node = tree.roots[0]
        assert orch_node.name == "orchestrator"
        assert len(orch_node.children) == 3

        for child in orch_node.children:
            assert child.status == "COMPLETED"
            assert child.kind == SpanKind.TOOL
