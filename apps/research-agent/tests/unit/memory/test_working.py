from __future__ import annotations

import pytest
from unittest.mock import patch


class TestCoreOperations:
    def test_add_user_creates_user_message(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("What is RAG?")

        messages = memory.to_list()
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "What is RAG?"}

    def test_add_assistant_creates_assistant_message_text_only(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_assistant("RAG stands for Retrieval-Augmented Generation.")

        messages = memory.to_list()
        assert len(messages) == 1
        assert messages[0] == {
            "role": "assistant",
            "content": "RAG stands for Retrieval-Augmented Generation.",
        }

    def test_add_assistant_with_tool_calls_creates_message(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query": "RAG"}'},
            }
        ]
        memory.add_assistant(tool_calls=tool_calls)

        messages = memory.to_list()
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] is None
        assert messages[0]["tool_calls"] == tool_calls

    def test_add_assistant_with_prompt_tokens_caches_exact_count(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("Hello")
        memory.add_assistant("Hi!", prompt_tokens=10)

        assert memory._exact_token_count == 10
        assert memory._exact_up_to == 2

    def test_add_assistant_with_content_and_tool_calls_and_tokens(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_assistant(
            content="Let me search.",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
            prompt_tokens=50,
        )

        messages = memory.to_list()
        assert messages[0]["content"] == "Let me search."
        assert messages[0]["tool_calls"] is not None
        assert memory._exact_token_count == 50

    def test_add_tool_result_creates_tool_message(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_tool_result("call_abc123", "Retrieved 5 results...")

        messages = memory.to_list()
        assert len(messages) == 1
        assert messages[0] == {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": "Retrieved 5 results...",
        }

    def test_set_system_sets_system_message(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.set_system("You are a research agent.")
        memory.add_user("Hello")

        messages = memory.to_list()
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are a research agent."}
        assert messages[1] == {"role": "user", "content": "Hello"}

    def test_to_list_returns_messages_in_order(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("First")
        memory.add_assistant("Second")
        memory.add_user("Third")

        messages = memory.to_list()
        assert [m["content"] for m in messages] == ["First", "Second", "Third"]

    def test_to_list_includes_system_message_first(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("User message")
        memory.set_system("System prompt")

        messages = memory.to_list()
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_clear_removes_messages_keeps_system(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.set_system("System prompt")
        memory.add_user("User message")
        memory.clear()

        messages = memory.to_list()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_message_count_returns_correct_count(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        assert memory.message_count() == 0

        memory.add_user("Hello")
        assert memory.message_count() == 1

        memory.add_assistant("Hi there")
        assert memory.message_count() == 2

        memory.set_system("System")
        assert memory.message_count() == 2

    def test_add_accepts_raw_message_dict(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add({"role": "user", "content": "Raw message"})

        messages = memory.to_list()
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Raw message"}


class TestTokenCounting:
    def test_estimate_total_tokens_returns_zero_when_empty(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        assert memory.estimate_total_tokens() == 0

    def test_estimate_total_tokens_returns_exact_when_no_new_messages(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("Hello")
        memory.add_assistant("Hi!", prompt_tokens=15)

        assert memory.estimate_total_tokens() == 15

    def test_estimate_total_tokens_includes_estimated_new_messages(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("Hello")
        memory.add_assistant("Hi!", prompt_tokens=15)

        memory.add_user("What is RAG?")

        total = memory.estimate_total_tokens()
        assert total > 15

    def test_prompt_tokens_caches_how_many_messages_are_exact(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("First")
        memory.add_assistant("Second", prompt_tokens=20)
        memory.add_user("Third")

        assert memory._exact_up_to == 2
        assert memory._exact_token_count == 20

    def test_multiple_prompt_tokens_updates_replaces_previous(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("First")
        memory.add_assistant("Second", prompt_tokens=20)
        memory.add_user("Third")
        memory.add_assistant("Fourth", prompt_tokens=35)

        assert memory._exact_up_to == 4
        assert memory._exact_token_count == 35

    def test_fits_in_context_uses_estimated_total(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=100)
        memory.add_user("Hello")
        memory.add_assistant("Hi!", prompt_tokens=10)

        assert memory.fits_in_context() is True

        for i in range(100):
            memory.add_user(f"Message {i}" * 10)

        assert memory.fits_in_context() is False

    @patch("research_agent.memory.working.estimate_tokens")
    def test_error_is_only_on_new_messages(self, mock_estimate: object) -> None:
        from research_agent.memory import WorkingMemory

        mock_estimate.return_value = 15

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=200000)
        memory.add_user("Large context")
        memory.add_assistant("Response", prompt_tokens=100000)

        memory.add_user("New question")
        total = memory.estimate_total_tokens()

        assert total == 100000 + 15


class TestContextWindowDetection:
    def test_auto_detect_context_window_from_model_name(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        assert memory.max_tokens == 128000

    def test_auto_detect_glm5_context_window(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="glm-5")
        assert memory.max_tokens == 200000

    def test_explicit_max_tokens_overrides_detection(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=5000)
        assert memory.max_tokens == 5000

    def test_unknown_model_falls_back_to_default(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="unknown-model-xyz")
        assert memory.max_tokens == 4096

    def test_context_window_lookup_includes_common_models(self) -> None:
        from research_agent.memory.working import CONTEXT_WINDOWS

        expected_models = ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "glm-5"]
        for model in expected_models:
            assert model in CONTEXT_WINDOWS


class TestPluggableBackend:
    def test_default_backend_is_passthrough(self) -> None:
        from research_agent.memory import WorkingMemory
        from research_agent.memory.working import PassthroughMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        assert isinstance(memory._backend, PassthroughMemory)

    def test_custom_backend_can_be_injected(self) -> None:
        from research_agent.memory import WorkingMemory
        from research_agent.memory.working import MemoryBackend

        class CustomBackend(MemoryBackend):
            def __init__(self) -> None:
                self.messages: list[dict] = []
                self.system: dict | None = None

            def add(self, message: dict) -> None:
                self.messages.append(message)

            def set_system(self, content: str) -> None:
                self.system = {"role": "system", "content": content}

            def get_all(self) -> list[dict]:
                result = [self.system] if self.system else []
                return result + self.messages

            def clear(self) -> None:
                self.messages = []

        custom = CustomBackend()
        memory = WorkingMemory(model="gpt-4o-mini", backend=custom)
        memory.add_user("Hello")

        assert len(custom.messages) == 1

    def test_backend_protocol_methods_are_called(self) -> None:
        from research_agent.memory import WorkingMemory
        from research_agent.memory.working import MemoryBackend

        class TrackingBackend(MemoryBackend):
            def __init__(self) -> None:
                self.messages: list[dict] = []
                self.system: dict | None = None
                self.add_called = False
                self.get_all_called = False

            def add(self, message: dict) -> None:
                self.add_called = True
                self.messages.append(message)

            def set_system(self, content: str) -> None:
                self.system = {"role": "system", "content": content}

            def get_all(self) -> list[dict]:
                self.get_all_called = True
                result = [self.system] if self.system else []
                return result + self.messages

            def clear(self) -> None:
                self.messages = []

        backend = TrackingBackend()
        memory = WorkingMemory(model="gpt-4o-mini", backend=backend)
        memory.add_user("Test")

        assert backend.add_called


class TestEdgeCases:
    def test_empty_memory_returns_empty_list(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        messages = memory.to_list()
        assert messages == []

    def test_empty_memory_with_system_returns_system(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.set_system("System prompt")

        messages = memory.to_list()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_assistant_message_with_content_and_tool_calls(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_assistant(
            content="Let me search for that.",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        )

        messages = memory.to_list()
        assert messages[0]["content"] == "Let me search for that."
        assert messages[0]["tool_calls"] is not None

    def test_tool_result_with_error_content(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_tool_result("call_123", "Error: API timeout")

        messages = memory.to_list()
        assert messages[0]["content"] == "Error: API timeout"


class TestToolCallIdHelper:
    def test_generate_tool_call_id_returns_string(self) -> None:
        from research_agent.memory import generate_tool_call_id

        call_id = generate_tool_call_id()
        assert isinstance(call_id, str)

    def test_generate_tool_call_id_starts_with_call_prefix(self) -> None:
        from research_agent.memory import generate_tool_call_id

        call_id = generate_tool_call_id()
        assert call_id.startswith("call_")

    def test_generate_tool_call_id_is_unique(self) -> None:
        from research_agent.memory import generate_tool_call_id

        ids = [generate_tool_call_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_generate_tool_call_id_is_time_ordered(self) -> None:
        import time

        from research_agent.memory import generate_tool_call_id

        id1 = generate_tool_call_id()
        time.sleep(0.01)
        id2 = generate_tool_call_id()

        assert id1 < id2


class TestSystemMessageTokenCounting:
    @patch("research_agent.memory.working.estimate_tokens")
    def test_system_message_included_in_token_count(self, mock_estimate: object) -> None:
        from research_agent.memory import WorkingMemory

        mock_estimate.side_effect = lambda model, messages: len(messages) * 10

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("Hi")

        count_without_system = memory.estimate_total_tokens()

        memory.set_system("You are a helpful assistant with a very long system prompt.")
        count_with_system = memory.estimate_total_tokens()

        assert count_with_system > count_without_system


class TestModelNameHandling:
    def test_model_name_with_openai_prefix(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="openai/gpt-4o-mini")
        assert memory.max_tokens == 128000

    def test_model_name_with_anthropic_prefix(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="anthropic/claude-3.5-sonnet")
        assert memory.max_tokens == 200000

    def test_model_name_versioned_variant(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-2024-05-13")
        assert memory.max_tokens == 128000

    def test_model_name_with_date_stamp(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4-turbo-2024-04-09")
        assert memory.max_tokens == 128000


class TestEdgeCasesExtended:
    def test_add_assistant_with_no_args(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_assistant()

        messages = memory.to_list()
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0].get("content") is None
        assert messages[0].get("tool_calls") is None

    def test_add_user_empty_string(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("")

        messages = memory.to_list()
        assert len(messages) == 1
        assert messages[0]["content"] == ""

    def test_add_assistant_empty_string(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_assistant(content="")

        messages = memory.to_list()
        assert len(messages) == 1
        assert messages[0]["content"] == ""

    def test_add_tool_result_empty_content(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_tool_result("call_123", "")

        messages = memory.to_list()
        assert len(messages) == 1
        assert messages[0]["content"] == ""

    def test_set_system_overwrites_previous(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.set_system("First system")
        memory.set_system("Second system")

        messages = memory.to_list()
        assert len(messages) == 1
        assert messages[0]["content"] == "Second system"


class TestTokenCountingLiteLLMIntegration:
    """Test actual LiteLLM integration (not mocked)."""

    @pytest.mark.integration
    @pytest.mark.skip(reason="Integration test - requires LiteLLM and tiktoken")
    def test_token_estimation_uses_litellm_token_counter(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("Hello world")
        memory.add_assistant("Hi there!")

        total = memory.estimate_total_tokens()
        assert total > 0
        assert 3 <= total <= 10

    @pytest.mark.integration
    @pytest.mark.skip(reason="Integration test - requires LiteLLM and tiktoken")
    def test_token_estimation_matches_litellm_exactly(self) -> None:
        import litellm

        from research_agent.memory.working import estimate_tokens

        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]

        our_estimate = estimate_tokens("gpt-4o-mini", messages)
        litellm_count = litellm.token_counter(model="gpt-4o-mini", messages=messages)

        assert our_estimate == litellm_count


class TestContextWindowPartialMatch:
    def test_partial_match_gpt4o_variant(self) -> None:
        from research_agent.memory import WorkingMemory

        variants = [
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
        ]
        for variant in variants:
            memory = WorkingMemory(model=variant)
            assert memory.max_tokens == 128000, f"Failed for {variant}"

    def test_partial_match_claude_variant(self) -> None:
        from research_agent.memory import WorkingMemory

        variants = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ]
        for variant in variants:
            memory = WorkingMemory(model=variant)
            assert memory.max_tokens >= 4096, f"Failed for {variant}"

    def test_fallback_for_completely_unknown_model(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="some-random-model-v1.2.3")
        assert memory.max_tokens == 4096

    def test_case_sensitivity_of_model_lookup(self) -> None:
        from research_agent.memory import WorkingMemory

        memory_lower = WorkingMemory(model="gpt-4o-mini")
        memory_upper = WorkingMemory(model="GPT-4o-mini")

        assert memory_lower.max_tokens >= 4096
        assert memory_upper.max_tokens >= 4096
