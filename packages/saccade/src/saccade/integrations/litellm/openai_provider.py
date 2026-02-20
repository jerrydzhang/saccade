"""OpenAI-compatible custom provider for LiteLLM.

Usage:
    from decimal import Decimal
    from saccade.integrations.litellm import (
        ModelPricing,
        OpenAICompatibleProvider,
        TracedLiteLLM,
    )

    provider = OpenAICompatibleProvider(
        provider_name="my-provider",
        api_base="https://api.example.com/v1",
        api_key="sk-xxx",
        default_model="llama-3.1-70b",
        default_pricing=ModelPricing(
            input_cost_per_token=Decimal("0.0000005"),
            output_cost_per_token=Decimal("0.0000015"),
        ),
    )

    llm = TracedLiteLLM(model="my-provider/llama-3.1-70b", provider=provider)

    async for chunk in llm.stream(messages=[{"role": "user", "content": "Hello"}]):
        print(chunk.delta, end="")
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, cast

import httpx
from litellm import CustomLLM
from litellm.types.utils import (
    Choices,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    Usage,
)

from .pricing import ModelPricing, RegisterableProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from decimal import Decimal


class OpenAICompatibleProvider(CustomLLM, RegisterableProvider):
    """OpenAI-compatible custom provider for LiteLLM with optional pricing."""

    def __init__(  # noqa: PLR0913
        self,
        provider_name: str,
        api_base: str,
        api_key: str,
        default_model: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        pricing: dict[str, ModelPricing] | None = None,
        default_pricing: ModelPricing | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.pricing = pricing or {}
        self.default_pricing = default_pricing

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    def _parse_model(self, model: str) -> str:
        prefix = f"{self.provider_name}/"
        if model.startswith(prefix):
            return model[len(prefix) :]
        return model

    def _build_response(self, data: dict[str, Any], model: str) -> ModelResponse:
        response = ModelResponse()
        response.id = data.get("id", "")
        response.created = data.get("created", int(time.time()))
        response.model = model
        response.object = data.get("object", "chat.completion")

        choices_data = data.get("choices", [])
        response.choices = [
            Choices(
                finish_reason=choice.get("finish_reason", "stop"),
                index=choice.get("index", i),
                message=Message(
                    content=choice.get("message", {}).get("content", ""),
                    role=choice.get("message", {}).get("role", "assistant"),
                    function_call=choice.get("message", {}).get("function_call"),
                    tool_calls=choice.get("message", {}).get("tool_calls"),
                ),
            )
            for i, choice in enumerate(choices_data)
        ]

        usage_data = data.get("usage", {})
        if usage_data:
            response.usage = Usage(  # type: ignore[attr-defined]
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                reasoning_tokens=usage_data.get("completion_tokens_details", {}).get(
                    "reasoning_tokens", 0
                ),
            )

        return response

    def _parse_streaming_chunk(self, data: dict[str, Any], index: int = 0) -> GenericStreamingChunk:
        choices = data.get("choices", [])
        choice = choices[0] if choices else {}

        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        is_finished = finish_reason is not None

        text = delta.get("content", "")

        tool_use: Any = None
        if delta.get("tool_calls"):
            tool_use = delta["tool_calls"]

        usage: Any = None
        if data.get("usage"):
            usage = {
                "completion_tokens": data["usage"].get("completion_tokens", 0),
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "total_tokens": data["usage"].get("total_tokens", 0),
            }

        return cast(
            "GenericStreamingChunk",
            {
                "finish_reason": finish_reason,
                "index": index,
                "is_finished": is_finished,
                "text": text,
                "tool_use": tool_use,
                "usage": usage,
            },
        )

    def completion(  # noqa: PLR0913
        self,
        model: str,
        messages: list[dict[str, Any]],
        model_response: ModelResponse,
        optional_params: dict[str, Any],
        litellm_params: dict[str, Any],
        headers: dict[str, str] | None,
        *_args: Any,
        **_kwargs: Any,
    ) -> ModelResponse:
        actual_model = self._parse_model(model) or self.default_model or model

        payload = {
            "model": actual_model,
            "messages": messages,
            **optional_params,
        }

        request_headers = {**self._get_headers(), **(headers or {})}

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.api_base}/chat/completions",
                headers=request_headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        return self._build_response(data, model)

    async def acompletion(  # noqa: PLR0913
        self,
        model: str,
        messages: list[dict[str, Any]],
        model_response: ModelResponse,
        optional_params: dict[str, Any],
        litellm_params: dict[str, Any],
        headers: dict[str, str] | None,
        *_args: Any,
        **_kwargs: Any,
    ) -> ModelResponse:
        actual_model = self._parse_model(model) or self.default_model or model

        payload = {
            "model": actual_model,
            "messages": messages,
            **optional_params,
        }

        request_headers = {**self._get_headers(), **(headers or {})}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_base}/chat/completions",
                headers=request_headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        return self._build_response(data, model)

    def streaming(  # noqa: PLR0913
        self,
        model: str,
        messages: list[dict[str, Any]],
        model_response: ModelResponse,
        optional_params: dict[str, Any],
        litellm_params: dict[str, Any],
        headers: dict[str, str] | None,
        *_args: Any,
        **_kwargs: Any,
    ) -> Iterator[GenericStreamingChunk]:
        actual_model = self._parse_model(model) or self.default_model or model

        payload = {
            "model": actual_model,
            "messages": messages,
            "stream": True,
            **optional_params,
        }

        request_headers = {**self._get_headers(), **(headers or {})}
        request_headers.pop("Content-Type", None)

        with (
            httpx.Client(timeout=self.timeout) as client,
            client.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                headers=request_headers,
                json=payload,
            ) as response,
        ):
            response.raise_for_status()

            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    yield self._parse_streaming_chunk(data)
                except json.JSONDecodeError:
                    continue

    async def astreaming(  # noqa: PLR0913
        self,
        model: str,
        messages: list[dict[str, Any]],
        model_response: ModelResponse,
        optional_params: dict[str, Any],
        litellm_params: dict[str, Any],
        headers: dict[str, str] | None,
        *_args: Any,
        **_kwargs: Any,
    ) -> AsyncIterator[GenericStreamingChunk]:
        actual_model = self._parse_model(model) or self.default_model or model

        payload = {
            "model": actual_model,
            "messages": messages,
            "stream": True,
            **optional_params,
        }

        request_headers = {**self._get_headers(), **(headers or {})}
        request_headers.pop("Content-Type", None)

        async with (
            httpx.AsyncClient(timeout=self.timeout) as client,
            client.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                headers=request_headers,
                json=payload,
            ) as response,
        ):
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    yield self._parse_streaming_chunk(data)
                except json.JSONDecodeError:
                    continue

    def register(self) -> None:
        import litellm

        litellm.custom_provider_map = [{"provider": self.provider_name, "custom_handler": self}]

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
    ) -> Decimal | None:
        actual_model = self._parse_model(model)

        if actual_model in self.pricing:
            return self.pricing[actual_model].calculate(
                input_tokens, output_tokens, reasoning_tokens
            )

        if self.default_pricing:
            return self.default_pricing.calculate(input_tokens, output_tokens, reasoning_tokens)

        return None


__all__ = ["OpenAICompatibleProvider"]
