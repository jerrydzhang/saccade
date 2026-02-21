from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saccade import EventType, Span, Trace

pytestmark = pytest.mark.unit


def make_stream_chunk(
    delta: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = delta
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].finish_reason = finish_reason

    if tool_calls:
        chunk.choices[0].delta.tool_calls = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc.get("id")
            mock_tc.type = tc.get("type", "function")
            mock_tc.function = MagicMock()
            mock_tc.function.name = tc.get("function", {}).get("name")
            mock_tc.function.arguments = tc.get("function", {}).get("arguments")
            chunk.choices[0].delta.tool_calls.append(mock_tc)

    chunk.usage = None
    if usage:
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = usage.get("prompt_tokens", 0)
        chunk.usage.completion_tokens = usage.get("completion_tokens", 0)
        chunk.usage.reasoning_tokens = usage.get("reasoning_tokens", 0)

    return chunk


def make_completion_response(
    content: str | None = "",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    usage: dict[str, Any] | None = None,
    response_cost: float | None = None,
) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = finish_reason

    if tool_calls:
        response.choices[0].message.tool_calls = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc.get("id")
            mock_tc.type = tc.get("type", "function")
            mock_tc.function = MagicMock()
            mock_tc.function.name = tc.get("function", {}).get("name")
            mock_tc.function.arguments = tc.get("function", {}).get("arguments")
            response.choices[0].message.tool_calls.append(mock_tc)

    response.usage = None
    if usage:
        response.usage = MagicMock()
        response.usage.prompt_tokens = usage.get("prompt_tokens", 0)
        response.usage.completion_tokens = usage.get("completion_tokens", 0)
        response.usage.reasoning_tokens = usage.get("reasoning_tokens", 0)

    response._hidden_params = {"response_cost": response_cost} if response_cost else {}

    return response


class TestToolCallParsing:
    @pytest.mark.asyncio
    async def test_stream_parses_tool_calls_deterministically(self):
        from saccade.integrations.litellm import TracedLiteLLM

        llm = TracedLiteLLM(model="test/model")

        chunks = [
            make_stream_chunk(delta=""),
            make_stream_chunk(
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"loc'},
                    }
                ]
            ),
            make_stream_chunk(
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": None, "arguments": 'ation": "SF"}'},
                    }
                ]
            ),
            make_stream_chunk(finish_reason="tool_calls"),
            make_stream_chunk(
                finish_reason="tool_calls",
                usage={"prompt_tokens": 50, "completion_tokens": 20},
            ),
        ]

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        mock_response = mock_stream()

        with patch("saccade.integrations.litellm.traced_llm.acompletion") as mock_acomplete:
            mock_acomplete.return_value = mock_response

            tool_calls_seen = []
            async for chunk in llm.stream(messages=[{"role": "user", "content": "test"}]):
                if chunk.tool_calls:
                    tool_calls_seen.extend(chunk.tool_calls)

        assert len(tool_calls_seen) == 2
        assert tool_calls_seen[0]["function"]["name"] == "get_weather"
        assert tool_calls_seen[0]["id"] == "call_123"

    @pytest.mark.asyncio
    async def test_complete_parses_tool_calls_deterministically(self):
        from saccade.integrations.litellm import TracedLiteLLM

        llm = TracedLiteLLM(model="test/model")

        mock_response = make_completion_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query": "test"}'},
                }
            ],
            finish_reason="tool_calls",
            usage={"prompt_tokens": 30, "completion_tokens": 15},
        )

        with patch("saccade.integrations.litellm.traced_llm.acompletion") as mock_acomplete:
            mock_acomplete.return_value = mock_response

            result = await llm.complete(messages=[{"role": "user", "content": "test"}])

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "search"
        assert result.tool_calls[0]["id"] == "call_456"
        assert result.finish_reason == "tool_calls"


class TestCostCalculation:
    @pytest.mark.asyncio
    async def test_stream_uses_litellm_cost_when_available(self):
        from saccade.integrations.litellm import TracedLiteLLM

        llm = TracedLiteLLM(model="test/model")

        chunks = [
            make_stream_chunk(delta="Hello"),
            make_stream_chunk(delta=" world"),
            make_stream_chunk(
                finish_reason="stop",
                usage={"prompt_tokens": 100, "completion_tokens": 10},
            ),
        ]

        class MockStreamResponse:
            _hidden_params = {"response_cost": 0.00234}

            def __init__(self, chunks):
                self._chunks = chunks
                self._index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._index]
                self._index += 1
                return chunk

        mock_response = MockStreamResponse(chunks)

        with Trace() as trace:
            with Span("llm", kind="llm"):
                with patch("saccade.integrations.litellm.traced_llm.acompletion") as mock_acomplete:
                    mock_acomplete.return_value = mock_response
                    async for _chunk in llm.stream(messages=[{"role": "user", "content": "test"}]):
                        pass

        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        assert len(success_events) == 1

        success = success_events[0]
        assert success.cost is not None
        assert success.cost.usd == Decimal("0.00234")

    @pytest.mark.asyncio
    async def test_complete_uses_litellm_cost_when_available(self):
        from saccade.integrations.litellm import TracedLiteLLM

        llm = TracedLiteLLM(model="test/model")

        mock_response = make_completion_response(
            content="Test response",
            usage={"prompt_tokens": 100, "completion_tokens": 10},
            response_cost=0.00150,
        )

        with Trace() as trace:
            with Span("llm", kind="llm"):
                with patch("saccade.integrations.litellm.traced_llm.acompletion") as mock_acomplete:
                    mock_acomplete.return_value = mock_response
                    await llm.complete(messages=[{"role": "user", "content": "test"}])

        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        success = success_events[0]

        assert success.cost is not None
        assert success.cost.usd == Decimal("0.00150")


class TestReasoningTokens:
    @pytest.mark.asyncio
    async def test_stream_captures_reasoning_tokens(self):
        from saccade.integrations.litellm import TracedLiteLLM

        llm = TracedLiteLLM(model="test/model")

        chunks = [
            make_stream_chunk(delta="Thinking"),
            make_stream_chunk(
                finish_reason="stop",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "reasoning_tokens": 200,
                },
            ),
        ]

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        mock_response = mock_stream()

        with Trace() as trace:
            with Span("llm", kind="llm"):
                with patch("saccade.integrations.litellm.traced_llm.acompletion") as mock_acomplete:
                    mock_acomplete.return_value = mock_response
                    async for _chunk in llm.stream(messages=[{"role": "user", "content": "test"}]):
                        pass

        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        success = success_events[0]

        assert success.tokens is not None
        assert success.tokens.input == 100
        assert success.tokens.output == 50
        assert success.tokens.reasoning == 200

    @pytest.mark.asyncio
    async def test_complete_captures_reasoning_tokens(self):
        from saccade.integrations.litellm import TracedLiteLLM

        llm = TracedLiteLLM(model="test/model")

        mock_response = make_completion_response(
            content="Response",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "reasoning_tokens": 150},
        )

        with Trace() as trace:
            with Span("llm", kind="llm"):
                with patch("saccade.integrations.litellm.traced_llm.acompletion") as mock_acomplete:
                    mock_acomplete.return_value = mock_response
                    result = await llm.complete(messages=[{"role": "user", "content": "test"}])

        assert result.usage is not None
        assert result.usage.reasoning_tokens == 150

        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        success = success_events[0]
        assert success.tokens is not None
        assert success.tokens.reasoning == 150


class TestStreamOptionsVerification:
    @pytest.mark.asyncio
    async def test_stream_includes_usage_option(self):
        from saccade.integrations.litellm import TracedLiteLLM

        llm = TracedLiteLLM(model="test/model")

        chunks = [make_stream_chunk(delta="Hi", finish_reason="stop")]

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        mock_response = mock_stream()

        with patch("saccade.integrations.litellm.traced_llm.acompletion") as mock_acomplete:
            mock_acomplete.return_value = mock_response

            async for _chunk in llm.stream(messages=[{"role": "user", "content": "test"}]):
                pass

            call_kwargs = mock_acomplete.call_args[1]
            assert "stream_options" in call_kwargs
            assert call_kwargs["stream_options"]["include_usage"] is True


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_stream_handles_api_error(self):
        from saccade.integrations.litellm import TracedLiteLLM

        llm = TracedLiteLLM(model="test/model")

        with patch("saccade.integrations.litellm.traced_llm.acompletion") as mock_acomplete:
            mock_acomplete.side_effect = Exception("API Error: Rate limited")

            with pytest.raises(Exception, match="API Error"):
                async for _chunk in llm.stream(messages=[{"role": "user", "content": "test"}]):
                    pass

    @pytest.mark.asyncio
    async def test_complete_handles_api_error(self):
        from saccade.integrations.litellm import TracedLiteLLM

        llm = TracedLiteLLM(model="test/model")

        with patch("saccade.integrations.litellm.traced_llm.acompletion") as mock_acomplete:
            mock_acomplete.side_effect = Exception("API Error: Invalid API key")

            with pytest.raises(Exception, match="API Error"):
                await llm.complete(messages=[{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_stream_works_without_usage(self):
        from saccade.integrations.litellm import TracedLiteLLM

        llm = TracedLiteLLM(model="test/model")

        chunks = [
            make_stream_chunk(delta="Hello"),
            make_stream_chunk(finish_reason="stop"),
        ]

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        mock_response = mock_stream()

        with Trace() as trace:
            with Span("llm", kind="llm"):
                with patch("saccade.integrations.litellm.traced_llm.acompletion") as mock_acomplete:
                    mock_acomplete.return_value = mock_response
                    chunks_received = []
                    async for chunk in llm.stream(messages=[{"role": "user", "content": "test"}]):
                        if chunk.delta:
                            chunks_received.append(chunk.delta)

        assert chunks_received == ["Hello"]

        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        success = success_events[0]
        assert success.cost is None

    @pytest.mark.asyncio
    async def test_complete_with_empty_content(self):
        from saccade.integrations.litellm import TracedLiteLLM

        llm = TracedLiteLLM(model="test/model")

        mock_response = make_completion_response(
            content="",
            usage={"prompt_tokens": 50, "completion_tokens": 0},
        )

        with Trace() as trace:
            with Span("llm", kind="llm"):
                with patch("saccade.integrations.litellm.traced_llm.acompletion") as mock_acomplete:
                    mock_acomplete.return_value = mock_response
                    result = await llm.complete(messages=[{"role": "user", "content": "test"}])

        assert result.content == ""
        assert result.usage is not None

        output_events = [e for e in trace.events if e.type == EventType.OUTPUT]
        assert len(output_events) == 0


class TestCostOverridePrecedence:
    @pytest.mark.asyncio
    async def test_cost_override_overrides_litellm_cost(self):
        from saccade.integrations.litellm import TracedLiteLLM

        llm = TracedLiteLLM(model="test/model")

        chunks = [
            make_stream_chunk(delta="Test"),
            make_stream_chunk(
                finish_reason="stop",
                usage={"prompt_tokens": 100, "completion_tokens": 10},
            ),
        ]

        async def mock_stream():
            response = MagicMock()
            response._hidden_params = {"response_cost": 0.001}
            for chunk in chunks:
                yield chunk

        mock_response = mock_stream()
        override_cost = Decimal("0.999")

        with Trace() as trace:
            with Span("llm", kind="llm"):
                with patch("saccade.integrations.litellm.traced_llm.acompletion") as mock_acomplete:
                    mock_acomplete.return_value = mock_response
                    async for _chunk in llm.stream(
                        messages=[{"role": "user", "content": "test"}],
                        cost_override=override_cost,
                    ):
                        pass

        success_events = [e for e in trace.events if e.type == EventType.SUCCESS]
        success = success_events[0]

        assert success.cost is not None
        assert success.cost.usd == override_cost
