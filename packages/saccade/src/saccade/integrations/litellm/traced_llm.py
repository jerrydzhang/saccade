"""LiteLLM wrapper with saccade tracing.

Usage:
    from saccade.integrations.litellm import TracedLiteLLM

    llm = TracedLiteLLM(model="openai/gpt-4o-mini")

    async for chunk in llm.stream(messages=[{"role": "user", "content": "Hello"}]):
        print(chunk.delta, end="")

    result = await llm.complete(messages=[{"role": "user", "content": "Hello"}])
    print(result.content)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from litellm import acompletion

from saccade import CostMetrics, OperationMeta, Span, TokenMetrics

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .pricing import RegisterableProvider


@dataclass(frozen=True)
class Usage:
    """Token usage from LLM response."""

    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class StreamChunk:
    """Incremental chunk from streaming completion."""

    delta: str
    tool_calls: list[dict[str, Any]] | None
    finish_reason: str | None
    usage: Usage | None = None
    _raw: Any = field(repr=False, default=None)


@dataclass(frozen=True)
class Completion:
    """Complete response from non-streaming completion."""

    content: str
    tool_calls: list[dict[str, Any]] | None
    finish_reason: str | None
    usage: Usage | None = None
    _raw: Any = field(repr=False, default=None)


class TracedLiteLLM:
    """LiteLLM wrapper with saccade tracing.

    Automatically emits CHUNK events during streaming and captures
    metrics (tokens, cost, latency) on span completion.
    """

    def __init__(
        self,
        model: str,
        provider: RegisterableProvider | None = None,
    ) -> None:
        self.model = model
        self._provider = provider
        if provider is not None:
            provider.register()

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        cost_override: Decimal | None = None,
    ) -> AsyncIterator[StreamChunk]:
        span = Span.current()
        content_accumulator: list[str] = []
        usage_data: Usage | None = None

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        response = await acompletion(**kwargs)

        async for chunk in response:
            choice = chunk.choices[0]
            delta = choice.delta

            delta_content = delta.content or ""

            tool_calls = None
            if delta.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id if hasattr(tc, "id") else None,
                        "type": tc.type if hasattr(tc, "type") else "function",
                        "function": {
                            "name": tc.function.name if tc.function else None,
                            "arguments": tc.function.arguments if tc.function else None,
                        },
                    }
                    for tc in delta.tool_calls
                ]

            chunk_finish = choice.finish_reason

            chunk_usage = None
            chunk_usage_data = getattr(chunk, "usage", None)
            if chunk_usage_data:
                reasoning = 0
                if (
                    hasattr(chunk_usage_data, "reasoning_tokens")
                    and chunk_usage_data.reasoning_tokens
                ):
                    reasoning = chunk_usage_data.reasoning_tokens

                chunk_usage = Usage(
                    input_tokens=chunk_usage_data.prompt_tokens,
                    output_tokens=chunk_usage_data.completion_tokens,
                    reasoning_tokens=reasoning,
                )
                usage_data = chunk_usage

            if delta_content and span:
                span.stream(delta_content)
                content_accumulator.append(delta_content)

            yield StreamChunk(
                delta=delta_content,
                tool_calls=tool_calls,
                finish_reason=chunk_finish,
                usage=chunk_usage,
                _raw=chunk,
            )

        if span:
            full_content = "".join(content_accumulator)
            if full_content:
                span.set_output(full_content)

            if usage_data:
                cost = cost_override
                if cost is None:
                    cost = self._extract_cost(response, usage_data)

                if cost is not None:
                    span.set_metrics(
                        tokens=TokenMetrics(
                            input=usage_data.input_tokens,
                            output=usage_data.output_tokens,
                            reasoning=usage_data.reasoning_tokens,
                        ),
                        cost=CostMetrics(usd=cost),
                    )
                else:
                    span.set_metrics(
                        tokens=TokenMetrics(
                            input=usage_data.input_tokens,
                            output=usage_data.output_tokens,
                            reasoning=usage_data.reasoning_tokens,
                        ),
                    )

                span.set_meta(
                    OperationMeta(
                        model=self.model,
                        kind="llm",
                    )
                )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        cost_override: Decimal | None = None,
    ) -> Completion:
        span = Span.current()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            kwargs["tools"] = tools

        response = await acompletion(**kwargs)

        choice = response.choices[0]
        message = choice.message

        content = message.content or ""

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id if hasattr(tc, "id") else None,
                    "type": tc.type if hasattr(tc, "type") else "function",
                    "function": {
                        "name": tc.function.name if tc.function else None,
                        "arguments": tc.function.arguments if tc.function else None,
                    },
                }
                for tc in message.tool_calls
            ]

        usage = None
        if response.usage:
            reasoning = 0
            if hasattr(response.usage, "reasoning_tokens") and response.usage.reasoning_tokens:
                reasoning = response.usage.reasoning_tokens

            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                reasoning_tokens=reasoning,
            )

        if span:
            if content:
                span.set_output(content)

            if usage:
                cost = cost_override
                if cost is None:
                    cost = self._extract_cost(response, usage)

                if cost is not None:
                    span.set_metrics(
                        tokens=TokenMetrics(
                            input=usage.input_tokens,
                            output=usage.output_tokens,
                            reasoning=usage.reasoning_tokens,
                        ),
                        cost=CostMetrics(usd=cost),
                    )
                else:
                    span.set_metrics(
                        tokens=TokenMetrics(
                            input=usage.input_tokens,
                            output=usage.output_tokens,
                            reasoning=usage.reasoning_tokens,
                        ),
                    )

                span.set_meta(
                    OperationMeta(
                        model=self.model,
                        kind="llm",
                    )
                )

        return Completion(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
            _raw=response,
        )

    def _extract_cost(self, response: Any, usage: Usage | None = None) -> Decimal | None:
        litellm_cost = self._extract_litellm_cost(response)
        if litellm_cost is not None:
            return litellm_cost

        provider_cost = self._extract_provider_cost(usage)
        if provider_cost is not None:
            return provider_cost

        return None

    def _extract_litellm_cost(self, response: Any) -> Decimal | None:
        try:
            hidden_params = getattr(response, "_hidden_params", {})
            if hidden_params and "response_cost" in hidden_params:
                cost = hidden_params["response_cost"]
                if cost is not None:
                    return Decimal(str(cost))
        except (AttributeError, TypeError, ValueError):
            pass
        return None

    def _extract_provider_cost(self, usage: Usage | None) -> Decimal | None:
        if self._provider and usage:
            return self._provider.calculate_cost(
                model=self.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            )
        return None


__all__ = ["Completion", "StreamChunk", "TracedLiteLLM", "Usage"]
