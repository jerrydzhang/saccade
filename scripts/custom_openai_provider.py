#!/usr/bin/env python3
from __future__ import annotations

"""
Custom OpenAI-Compatible Provider for LiteLLM

Creates a first-class custom provider that uses OpenAI's API format.
This gives you:
- Clean provider name in logs (e.g., "my-provider" not "openai")
- Proper tracing/metrics attribution
- Custom auth headers if needed

Usage:
    1. Edit CONFIG section below
    2. uv pip install -e "packages/saccade[litellm]"
    3. uv run scripts/custom_openai_provider.py
"""

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================

PROVIDER_NAME = "my-provider"
API_BASE = "https://api.z.ai/api/coding/paas/v4"
API_KEY = ""
DEFAULT_MODEL = "glm-4.7"

# =============================================================================

import time
from typing import Any

import httpx
from litellm import CustomLLM, completion
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    Usage,
)


class OpenAICompatibleProvider(CustomLLM):
    """
    A proper custom provider that speaks OpenAI's API format.

    Usage:
        provider = OpenAICompatibleProvider(
            provider_name="my-provider",
            api_base="https://api.example.com/v1",
            api_key="sk-xxx",
            default_model="llama-3.1-70b",
        )

        # Register globally
        provider.register()

        # Then use anywhere
        response = completion(
            model="my-provider/llama-3.1-70b",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    def __init__(
        self,
        provider_name: str,
        api_base: str,
        api_key: str,
        default_model: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
        self.extra_headers = extra_headers or {}

    def _get_headers(self) -> dict[str, str]:
        """Build request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    def _parse_model(self, model: str) -> str:
        """Extract model name, removing provider prefix if present."""
        prefix = f"{self.provider_name}/"
        if model.startswith(prefix):
            return model[len(prefix) :]
        return model

    def _build_response(self, data: dict[str, Any], model: str) -> ModelResponse:
        """Transform API response to LiteLLM ModelResponse."""
        response = ModelResponse()
        response.id = data.get("id", "")
        response.created = data.get("created", int(time.time()))
        response.model = model
        response.object = data.get("object", "chat.completion")

        # Parse choices
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

        # Parse usage
        usage_data = data.get("usage", {})
        if usage_data:
            response.usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                reasoning_tokens=usage_data.get("completion_tokens_details", {}).get(
                    "reasoning_tokens", 0
                ),
            )

        return response

    def completion(  # noqa: PLR0913, PLR0917
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
        """Synchronous completion."""
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

    async def acompletion(  # noqa: PLR0913, PLR0917
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
        """Async completion."""
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

    # Streaming support can be added by implementing streaming() method
    # See: https://docs.litellm.ai/docs/providers/custom_llm_server

    def register(self) -> None:
        """Register this provider globally with LiteLLM."""
        import litellm

        litellm.custom_provider_map = [
            {"provider": self.provider_name, "custom_handler": self}
        ]


# =============================================================================
# USAGE EXAMPLE
# =============================================================================


def main() -> None:
    """Demo the custom provider."""
    print("=" * 60)
    print("Custom OpenAI-Compatible Provider")
    print("=" * 60)

    # Create provider
    provider = OpenAICompatibleProvider(
        provider_name=PROVIDER_NAME,
        api_base=API_BASE,
        api_key=API_KEY,
        default_model=DEFAULT_MODEL,
    )

    # Register globally
    provider.register()
    print(f"\n✓ Registered provider: {PROVIDER_NAME}")
    print(f"  API Base: {API_BASE}")
    print(f"  Default Model: {DEFAULT_MODEL}")

    # Now you can use it anywhere with clean provider attribution
    print("\n--- Usage Examples ---")

    usage_code = f'''
from litellm import completion

# Sync
response = completion(
    model="{PROVIDER_NAME}/{DEFAULT_MODEL}",
    messages=[{{"role": "user", "content": "Hello!"}}],
)

# Async
import litellm
response = await litellm.acompletion(
    model="{PROVIDER_NAME}/{DEFAULT_MODEL}",
    messages=[{{"role": "user", "content": "Hello!"}}],
)
'''
    print(usage_code)

    # Test with actual API (uncomment after configuring):
    response = completion(
        model=f"{PROVIDER_NAME}/{DEFAULT_MODEL}",
        messages=[{"role": "user", "content": "Say hello"}],
    )
    print(f"Response: {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
