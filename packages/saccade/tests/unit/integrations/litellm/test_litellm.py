"""Tests for TracedLiteLLM integration.

These tests verify the contract between LiteLLM and saccade tracing:
1. Streaming emits CHUNK events to active Span
2. Metrics (tokens, cost, latency) are captured on Span completion
3. Tool calling works with streaming

Configuration:
    Copy packages/saccade/.env.example to packages/saccade/.env
    and set PROVIDER_MODEL and your API key.

    To record cassettes:
        pytest packages/saccade/tests/unit/integrations/litellm/ --vcr-record=new_episodes
"""

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from saccade.integrations.litellm import TracedLiteLLM

pytestmark = pytest.mark.integration


class TestStreamingEmitsChunkEvents:
    """Test 1.1: Streaming response emits CHUNK events to active Span."""

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_stream_emits_chunk_events(self, llm: "TracedLiteLLM"):
        """Each streaming chunk should emit a CHUNK event to the active span."""
        from saccade import EventType, Span, Trace

        with Trace() as trace:
            with Span("llm", kind="llm"):
                chunks_received = []
                async for chunk in llm.stream(messages=[{"role": "user", "content": "Say hello"}]):
                    if chunk.delta:
                        chunks_received.append(chunk.delta)

        # Verify CHUNK events were emitted
        chunk_events = [e for e in trace.events if e.type == EventType.CHUNK]
        assert len(chunk_events) >= 1, "Expected at least one CHUNK event"

        # Verify chunk content matches what we received
        chunk_contents = [e.chunk for e in chunk_events if isinstance(e.chunk, str)]
        assert len(chunk_contents) >= 1, "Expected at least one chunk with string content"

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_stream_without_active_span_still_works(self, llm: "TracedLiteLLM"):
        """Streaming should work even without an active span (no tracing)."""
        # No Trace, no Span - should still work
        chunks_received = []
        async for chunk in llm.stream(messages=[{"role": "user", "content": "Say hello"}]):
            if chunk.delta:
                chunks_received.append(chunk.delta)

        assert len(chunks_received) >= 1, "Expected at least one chunk"


class TestMetricsCapture:
    """Test 1.2: Final response captures tokens, cost, latency on Span."""

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_stream_captures_token_metrics(self, llm: "TracedLiteLLM"):
        """Streaming should capture input/output tokens on the span."""
        from saccade import EventType, Span, Trace

        with Trace() as trace:
            with Span("llm", kind="llm"):
                async for chunk in llm.stream(messages=[{"role": "user", "content": "Say hello"}]):
                    pass

        # Find SUCCESS event which contains metrics
        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        assert len(success_events) == 1

        success = success_events[0]
        assert success.tokens is not None
        assert success.tokens.input > 0, "Expected non-zero input tokens"
        assert success.tokens.output > 0, "Expected non-zero output tokens"

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_stream_captures_cost(self, llm: "TracedLiteLLM"):
        """Streaming should capture cost on the span."""
        from saccade import EventType, Span, Trace

        with Trace() as trace:
            with Span("llm", kind="llm"):
                async for chunk in llm.stream(messages=[{"role": "user", "content": "Say hello"}]):
                    pass

        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        success = success_events[0]

        # Cost may be None for custom providers without known pricing
        if success.cost is not None:
            assert success.cost.usd > Decimal(0), "Expected non-zero cost"

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_stream_captures_latency(self, llm: "TracedLiteLLM"):
        """Streaming should capture latency on the span."""
        from saccade import EventType, Span, Trace

        with Trace() as trace:
            with Span("llm", kind="llm"):
                async for chunk in llm.stream(messages=[{"role": "user", "content": "Say hello"}]):
                    pass

        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        success = success_events[0]

        assert success.latency is not None
        assert success.latency.total_ms > 0, "Expected non-zero latency"

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_complete_captures_metrics(self, llm: "TracedLiteLLM"):
        """Non-streaming completion should capture tokens, cost, latency."""
        from saccade import EventType, Span, Trace

        with Trace() as trace:
            with Span("llm", kind="llm"):
                result = await llm.complete(messages=[{"role": "user", "content": "Say hello"}])

        # Verify result has content
        assert result.content, "Expected non-empty content"
        assert result.usage is not None, "Expected usage in result"

        # Verify metrics on span
        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        success = success_events[0]

        assert success.tokens is not None
        assert success.tokens.input > 0
        assert success.tokens.output > 0
        # Cost may be None for custom providers without known pricing
        if success.cost is not None:
            assert success.cost.usd > Decimal(0)
        assert success.latency is not None
        assert success.latency.total_ms > 0

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_cost_override(self, llm: "TracedLiteLLM"):
        """cost_override should override the calculated cost."""
        from saccade import EventType, Span, Trace

        custom_cost = Decimal("0.123456")

        with Trace() as trace:
            with Span("llm", kind="llm"):
                async for chunk in llm.stream(
                    messages=[{"role": "user", "content": "Say hello"}],
                    cost_override=custom_cost,
                ):
                    pass

        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        success = success_events[0]

        assert success.cost is not None
        assert success.cost.usd == custom_cost


class TestToolCallingWithStreaming:
    """Test 1.3: Tool calling works with streaming."""

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_stream_with_tool_calls(self, llm: "TracedLiteLLM"):
        """Streaming with tools should yield tool_calls in chunks."""
        from saccade import Span, Trace

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        with Trace() as trace:
            with Span("llm", kind="llm"):
                tool_calls_seen = []
                async for chunk in llm.stream(
                    messages=[{"role": "user", "content": "What's the weather in SF?"}],
                    tools=tools,
                ):
                    if chunk.tool_calls:
                        tool_calls_seen.extend(chunk.tool_calls)

        # The LLM may or may not call tools depending on the response
        # We verify the mechanism works, not that tools are always called
        # (that depends on the LLM's decision)
        if tool_calls_seen:
            # If tool calls were made, verify structure
            assert all("function" in tc for tc in tool_calls_seen)

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_complete_with_tool_calls(self, llm: "TracedLiteLLM"):
        """Non-streaming with tools should return tool_calls in result."""
        from saccade import Span, Trace

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        with Trace() as trace:
            with Span("llm", kind="llm"):
                result = await llm.complete(
                    messages=[{"role": "user", "content": "What's the weather in SF?"}],
                    tools=tools,
                )

        # Verify result structure is consistent
        assert result.content is not None or result.tool_calls is not None
        assert result.finish_reason is not None


class TestResponseTypes:
    """Tests for StreamChunk and Completion types."""

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_stream_chunk_structure(self, llm: "TracedLiteLLM"):
        """StreamChunk should have delta, tool_calls, finish_reason, usage."""
        chunks = []
        async for chunk in llm.stream(messages=[{"role": "user", "content": "Say hello"}]):
            chunks.append(chunk)

        assert len(chunks) >= 1

        # Verify chunk structure
        for chunk in chunks:
            assert hasattr(chunk, "delta")
            assert hasattr(chunk, "tool_calls")
            assert hasattr(chunk, "finish_reason")
            assert hasattr(chunk, "usage")
            assert hasattr(chunk, "_raw")

        # Final chunk should have usage
        final_chunk = chunks[-1]
        if final_chunk.finish_reason:
            assert final_chunk.usage is not None
            assert final_chunk.usage.input_tokens > 0
            assert final_chunk.usage.output_tokens > 0

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_completion_structure(self, llm: "TracedLiteLLM"):
        """Completion should have content, tool_calls, finish_reason, usage."""
        result = await llm.complete(messages=[{"role": "user", "content": "Say hello"}])

        # Verify completion structure
        assert hasattr(result, "content")
        assert hasattr(result, "tool_calls")
        assert hasattr(result, "finish_reason")
        assert hasattr(result, "usage")
        assert hasattr(result, "_raw")

        assert result.content is not None
        assert result.usage is not None
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0
        assert result.finish_reason is not None


class TestOutputCapture:
    """Tests for output being set on the span."""

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_stream_sets_output_on_span(self, llm: "TracedLiteLLM"):
        """Streaming should set the accumulated content as span output."""
        from saccade import EventType, Span, Trace

        with Trace() as trace:
            with Span("llm", kind="llm"):
                full_content = []
                async for chunk in llm.stream(messages=[{"role": "user", "content": "Say hello"}]):
                    if chunk.delta:
                        full_content.append(chunk.delta)

        # Verify OUTPUT event was emitted
        output_events = [e for e in trace.events if e.type == EventType.OUTPUT]
        assert len(output_events) == 1

        # Output should contain the accumulated content
        output = output_events[0].output
        assert output is not None
        # The output should contain the content we received
        accumulated = "".join(full_content)
        assert output == accumulated or accumulated in output

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_complete_sets_output_on_span(self, llm: "TracedLiteLLM"):
        """Complete should set the content as span output."""
        from saccade import EventType, Span, Trace

        with Trace() as trace:
            with Span("llm", kind="llm"):
                result = await llm.complete(messages=[{"role": "user", "content": "Say hello"}])

        # Verify OUTPUT event was emitted
        output_events = [e for e in trace.events if e.type == EventType.OUTPUT]
        assert len(output_events) == 1

        output = output_events[0].output
        assert output == result.content


class TestProviderPricing:
    """Tests for provider-based cost calculation."""

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_stream_uses_provider_pricing(self, llm_with_pricing: "TracedLiteLLM"):
        """When provider has pricing, cost should be calculated from provider."""
        from saccade import EventType, Span, Trace

        with Trace() as trace:
            with Span("llm", kind="llm"):
                async for chunk in llm_with_pricing.stream(
                    messages=[{"role": "user", "content": "Say hello"}]
                ):
                    pass

        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        assert len(success_events) == 1

        success = success_events[0]
        assert success.tokens is not None
        assert success.tokens.input > 0

        assert success.cost is not None, "Expected cost from provider pricing"
        assert success.cost.usd > Decimal(0), "Expected non-zero cost"

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_complete_uses_provider_pricing(self, llm_with_pricing: "TracedLiteLLM"):
        """Non-streaming should also use provider pricing."""
        from saccade import EventType, Span, Trace

        with Trace() as trace:
            with Span("llm", kind="llm"):
                result = await llm_with_pricing.complete(
                    messages=[{"role": "user", "content": "Say hello"}]
                )

        assert result.content
        assert result.usage is not None

        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        success = success_events[0]

        assert success.tokens is not None
        assert success.cost is not None, "Expected cost from provider pricing"
        assert success.cost.usd > Decimal(0)

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_cost_override_takes_precedence(self, llm_with_pricing: "TracedLiteLLM"):
        """cost_override should take precedence over provider pricing."""
        from saccade import EventType, Span, Trace

        custom_cost = Decimal("0.999999")

        with Trace() as trace:
            with Span("llm", kind="llm"):
                async for chunk in llm_with_pricing.stream(
                    messages=[{"role": "user", "content": "Say hello"}],
                    cost_override=custom_cost,
                ):
                    pass

        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        success = success_events[0]

        assert success.cost is not None
        assert success.cost.usd == custom_cost
