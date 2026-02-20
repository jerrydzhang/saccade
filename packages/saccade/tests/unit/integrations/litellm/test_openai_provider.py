"""Tests for OpenAICompatibleProvider.

These tests verify the provider correctly:
1. Transforms requests/responses to/from OpenAI format
2. Handles streaming via SSE
3. Registers with LiteLLM
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from saccade.integrations.litellm import OpenAICompatibleProvider

pytestmark = pytest.mark.unit


class TestProviderInitialization:
    def test_init_strips_trailing_slash_from_api_base(self):
        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_base="https://api.example.com/v1/",
            api_key="test-key",
        )
        assert provider.api_base == "https://api.example.com/v1"

    def test_init_stores_extra_headers(self):
        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_base="https://api.example.com/v1",
            api_key="test-key",
            extra_headers={"X-Custom": "value"},
        )
        assert provider.extra_headers == {"X-Custom": "value"}


class TestModelParsing:
    def test_parse_model_with_provider_prefix(self):
        provider = OpenAICompatibleProvider(
            provider_name="my-provider",
            api_base="https://api.example.com/v1",
            api_key="test-key",
        )
        assert provider._parse_model("my-provider/glm-4.7") == "glm-4.7"

    def test_parse_model_without_prefix(self):
        provider = OpenAICompatibleProvider(
            provider_name="my-provider",
            api_base="https://api.example.com/v1",
            api_key="test-key",
        )
        assert provider._parse_model("glm-4.7") == "glm-4.7"


class TestBuildResponse:
    def test_build_response_extracts_content(self):
        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_base="https://api.example.com/v1",
            api_key="test-key",
        )
        data = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {"content": "Hello world", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        response = provider._build_response(data, "test/model")
        assert response.choices[0].message.content == "Hello world"
        assert response.choices[0].finish_reason == "stop"

    def test_build_response_extracts_usage(self):
        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_base="https://api.example.com/v1",
            api_key="test-key",
        )
        data = {
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        response = provider._build_response(data, "test/model")
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

    def test_build_response_handles_tool_calls(self):
        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_base="https://api.example.com/v1",
            api_key="test-key",
        )
        data = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        response = provider._build_response(data, "test/model")
        assert response.choices[0].message.tool_calls is not None


class TestParseStreamingChunk:
    def test_parse_chunk_extracts_text(self):
        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_base="https://api.example.com/v1",
            api_key="test-key",
        )
        data = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
        chunk = provider._parse_streaming_chunk(data)
        assert chunk["text"] == "Hello"
        assert chunk["is_finished"] is False

    def test_parse_chunk_detects_finish(self):
        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_base="https://api.example.com/v1",
            api_key="test-key",
        )
        data = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        chunk = provider._parse_streaming_chunk(data)
        assert chunk["is_finished"] is True
        assert chunk["finish_reason"] == "stop"

    def test_parse_chunk_extracts_usage(self):
        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_base="https://api.example.com/v1",
            api_key="test-key",
        )
        data = {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        chunk = provider._parse_streaming_chunk(data)
        assert chunk["usage"]["prompt_tokens"] == 10


class TestStreaming:
    @pytest.mark.asyncio
    async def test_astreaming_yields_chunks(self):
        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_base="https://api.example.com/v1",
            api_key="test-key",
        )

        sse_lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.raise_for_status = MagicMock()

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        with patch.object(httpx, "AsyncClient", return_value=mock_client):
            chunks = []
            async for chunk in provider.astreaming(
                model="test/model",
                messages=[{"role": "user", "content": "Hi"}],
                model_response=MagicMock(),
                optional_params={},
                litellm_params={},
                headers=None,
            ):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0]["text"] == "Hello"
        assert chunks[1]["text"] == " world"


class TestRegistration:
    def test_register_updates_litellm_provider_map(self):
        provider = OpenAICompatibleProvider(
            provider_name="my-provider",
            api_base="https://api.example.com/v1",
            api_key="test-key",
        )

        with patch("litellm.custom_provider_map", []):
            provider.register()
            import litellm

            assert len(litellm.custom_provider_map) == 1
            assert litellm.custom_provider_map[0]["provider"] == "my-provider"
            assert litellm.custom_provider_map[0]["custom_handler"] is provider


async def aiter(items):
    for item in items:
        yield item
