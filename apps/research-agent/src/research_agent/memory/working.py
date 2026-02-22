"""Working memory implementation for research agent conversations.

This module provides token-aware message storage with pluggable backends.
Auto-truncation is deferred to a future phase - currently stores all messages
and relies on the agent to handle context limits.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import ulid


CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "claude-3.5-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-haiku": 200000,
    "glm-4": 128000,
    "glm-5": 200000,
    "_default": 4096,
}


def _lookup_context_window(model: str) -> int:
    if "/" in model:
        model = model.split("/", 1)[1]

    model_lower = model.lower()

    if model_lower in CONTEXT_WINDOWS:
        return CONTEXT_WINDOWS[model_lower]

    normalized = model_lower.replace(".", "-")
    if normalized in CONTEXT_WINDOWS:
        return CONTEXT_WINDOWS[normalized]

    for key in CONTEXT_WINDOWS:
        if model_lower.startswith(key) or normalized.startswith(key):
            return CONTEXT_WINDOWS[key]

    for key in CONTEXT_WINDOWS:
        key_normalized = key.replace(".", "-")
        if model_lower.startswith(key_normalized):
            return CONTEXT_WINDOWS[key]

    return CONTEXT_WINDOWS["_default"]


def estimate_tokens(model: str, messages: list[dict]) -> int:
    import litellm

    return litellm.token_counter(model=model, messages=messages)


def generate_tool_call_id() -> str:
    return f"call_{ulid.ULID()}"


@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol for pluggable memory backends."""

    def add(self, message: dict) -> None: ...
    def set_system(self, content: str) -> None: ...
    def get_all(self) -> list[dict]: ...
    def clear(self) -> None: ...


class PassthroughMemory:
    """Simple memory backend that stores all messages without truncation."""

    def __init__(self) -> None:
        self._messages: list[dict] = []
        self._system: dict | None = None

    def add(self, message: dict) -> None:
        self._messages.append(message)

    def set_system(self, content: str) -> None:
        self._system = {"role": "system", "content": content}

    def get_all(self) -> list[dict]:
        if self._system:
            return [self._system] + self._messages
        return list(self._messages)

    def clear(self) -> None:
        self._messages = []


class WorkingMemory:
    """Token-aware conversation memory with hybrid counting.

    Stores message history with:
    - Exact token counts from API responses (prompt_tokens)
    - Estimated counts for new messages
    - Context window detection for fits_in_context() checks

    Note: Auto-truncation is deferred to a future phase. The agent is
    responsible for handling context limit exceeded scenarios.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int | None = None,
        backend: MemoryBackend | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens or _lookup_context_window(model)
        self._backend: MemoryBackend = backend or PassthroughMemory()
        self._exact_token_count: int = 0
        self._exact_up_to: int = 0

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def _backend(self) -> MemoryBackend:
        return self.__backend

    @_backend.setter
    def _backend(self, value: MemoryBackend) -> None:
        self.__backend = value

    def set_system(self, content: str) -> None:
        self._backend.set_system(content)

    def add_user(self, content: str) -> None:
        self._backend.add({"role": "user", "content": content})

    def add_assistant(
        self,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
        prompt_tokens: int | None = None,
    ) -> None:
        message: dict = {"role": "assistant"}

        if content is not None:
            message["content"] = content
        else:
            message["content"] = None

        if tool_calls is not None:
            message["tool_calls"] = tool_calls

        self._backend.add(message)

        if prompt_tokens is not None:
            all_messages = self._backend.get_all()
            self._exact_token_count = prompt_tokens
            self._exact_up_to = len(all_messages)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._backend.add(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    def add(self, message: dict) -> None:
        self._backend.add(message)

    def to_list(self) -> list[dict]:
        return self._backend.get_all()

    def clear(self) -> None:
        self._backend.clear()
        self._exact_token_count = 0
        self._exact_up_to = 0

    def message_count(self) -> int:
        all_messages = self._backend.get_all()
        return sum(1 for m in all_messages if m.get("role") != "system")

    def estimate_total_tokens(self) -> int:
        all_messages = self._backend.get_all()

        if not all_messages:
            return 0

        if self._exact_up_to >= len(all_messages):
            return self._exact_token_count

        if self._exact_up_to > 0:
            new_messages = all_messages[self._exact_up_to :]

            if new_messages:
                new_estimated = estimate_tokens(self._model, new_messages)
                return self._exact_token_count + new_estimated

            return self._exact_token_count

        return estimate_tokens(self._model, all_messages)

    def fits_in_context(self) -> bool:
        return self.estimate_total_tokens() <= self._max_tokens
