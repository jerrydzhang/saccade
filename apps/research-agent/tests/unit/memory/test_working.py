from __future__ import annotations

from unittest.mock import patch


class TestCoreOperations:
    def test_add_user_creates_user_message(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("What is RAG?")

        messages = memory.to_list(truncate=False)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "What is RAG?"}

    def test_add_assistant_creates_assistant_message_text_only(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_assistant("RAG stands for Retrieval-Augmented Generation.")

        messages = memory.to_list(truncate=False)
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

        messages = memory.to_list(truncate=False)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] is None
        assert messages[0]["tool_calls"] == tool_calls

    def test_add_assistant_with_prompt_tokens_caches_exact_count(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("Hello")
        memory.add_assistant("Hi!", prompt_tokens=10)

        # The exact count should be cached
        assert memory._exact_token_count == 10
        assert memory._exact_up_to == 2  # Both messages are now exact

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

        messages = memory.to_list(truncate=False)
        assert messages[0]["content"] == "Let me search."
        assert messages[0]["tool_calls"] is not None
        assert memory._exact_token_count == 50

    def test_add_tool_result_creates_tool_message(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_tool_result("call_abc123", "Retrieved 5 results...")

        messages = memory.to_list(truncate=False)
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

        messages = memory.to_list(truncate=False)
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are a research agent."}
        assert messages[1] == {"role": "user", "content": "Hello"}

    def test_to_list_returns_messages_in_order(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("First")
        memory.add_assistant("Second")
        memory.add_user("Third")

        messages = memory.to_list(truncate=False)
        assert [m["content"] for m in messages] == ["First", "Second", "Third"]

    def test_to_list_includes_system_message_first(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("User message")
        memory.set_system("System prompt")

        messages = memory.to_list(truncate=False)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_clear_removes_messages_keeps_system(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.set_system("System prompt")
        memory.add_user("User message")
        memory.clear()

        messages = memory.to_list(truncate=False)
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

        messages = memory.to_list(truncate=False)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Raw message"}


class TestTokenCounting:
    def test_estimate_total_tokens_returns_zero_when_empty(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        assert memory.estimate_total_tokens() == 0

    def test_estimate_total_tokens_returns_exact_when_no_new_messages(self) -> None:
        """After caching prompt_tokens, estimate should return exact count."""
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("Hello")
        memory.add_assistant("Hi!", prompt_tokens=15)

        # No new messages, so estimate should be exactly what we cached
        assert memory.estimate_total_tokens() == 15

    def test_estimate_total_tokens_includes_estimated_new_messages(self) -> None:
        """After adding new messages, estimate = exact + estimated new."""
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("Hello")
        memory.add_assistant("Hi!", prompt_tokens=15)

        # Add new message - should be estimated
        memory.add_user("What is RAG?")

        total = memory.estimate_total_tokens()
        # Total = 15 (exact) + estimated new message
        assert total > 15

    def test_prompt_tokens_caches_how_many_messages_are_exact(self) -> None:
        """prompt_tokens tracks both count and which messages are confirmed."""
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("First")
        memory.add_assistant("Second", prompt_tokens=20)
        memory.add_user("Third")  # New, not exact

        # Only first 2 messages are exact
        assert memory._exact_up_to == 2
        assert memory._exact_token_count == 20

    def test_multiple_prompt_tokens_updates_replaces_previous(self) -> None:
        """Each prompt_tokens call replaces the previous exact count."""
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_user("First")
        memory.add_assistant("Second", prompt_tokens=20)
        memory.add_user("Third")
        memory.add_assistant("Fourth", prompt_tokens=35)

        # Now 4 messages are exact with count 35
        assert memory._exact_up_to == 4
        assert memory._exact_token_count == 35

    def test_fits_in_context_uses_estimated_total(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=100)
        memory.add_user("Hello")
        memory.add_assistant("Hi!", prompt_tokens=10)

        assert memory.fits_in_context() is True

        # Add messages until over limit
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

        exact_token_count = 100000
        estimated_new_tokens = 15

        memory.add_user("New question")
        total = memory.estimate_total_tokens()

        assert total == exact_token_count + estimated_new_tokens


class TestTruncation:
    @patch("research_agent.memory.working.estimate_tokens")
    def test_to_list_truncate_true_truncates_to_max_tokens(self, mock_estimate: object) -> None:
        from research_agent.memory import WorkingMemory

        mock_estimate.side_effect = lambda model, messages: len(messages) * 10

        max_tokens = 20
        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=max_tokens)
        for i in range(10):
            memory.add_user(f"Message number {i}")

        messages = memory.to_list(truncate=True)

        assert len(messages) <= 2

    @patch("research_agent.memory.working.estimate_tokens")
    def test_truncation_preserves_system_message(self, mock_estimate: object) -> None:
        from research_agent.memory import WorkingMemory

        mock_estimate.side_effect = lambda model, messages: len(messages) * 10

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=20)
        memory.set_system("You are a helpful assistant.")
        for i in range(10):
            memory.add_user(f"Message number {i}")

        messages = memory.to_list(truncate=True)
        assert messages[0]["role"] == "system"

    @patch("research_agent.memory.working.estimate_tokens")
    def test_truncation_keeps_most_recent_messages(self, mock_estimate: object) -> None:
        from research_agent.memory import WorkingMemory

        mock_estimate.side_effect = lambda model, messages: len(messages) * 10

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=25)
        memory.add_user("First message")
        memory.add_user("Second message")
        memory.add_user("Third message")
        memory.add_user("Fourth message")

        messages = memory.to_list(truncate=True)
        contents = [m["content"] for m in messages if m["role"] == "user"]

        assert "Fourth message" in contents
        assert "First message" not in contents

    @patch("research_agent.memory.working.estimate_tokens")
    def test_truncation_with_keep_recent_minimum(self, mock_estimate: object) -> None:
        from research_agent.memory import WorkingMemory

        mock_estimate.side_effect = lambda model, messages: len(messages) * 10

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=30, keep_recent=2)
        for i in range(10):
            memory.add_user(f"Message {i}")

        messages = memory.to_list(truncate=True)
        user_messages = [m for m in messages if m["role"] == "user"]

        assert len(user_messages) >= 2

    def test_to_list_truncate_false_returns_all(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=10)
        for i in range(5):
            memory.add_user(f"Message {i}")

        messages = memory.to_list(truncate=False)
        assert len(messages) == 5

    @patch("research_agent.memory.working.estimate_tokens")
    def test_truncation_resets_exact_count(self, mock_estimate: object) -> None:
        from research_agent.memory import WorkingMemory

        mock_estimate.side_effect = lambda model, messages: len(messages) * 10

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=20)
        memory.add_user("First")
        memory.add_assistant("Second", prompt_tokens=100)
        memory.add_user("Third")

        memory.to_list(truncate=True)

        assert memory._exact_token_count == 0
        assert memory._exact_up_to == 0


class TestToolPairPreservation:
    @patch("research_agent.memory.working.estimate_tokens")
    def test_truncation_does_not_split_tool_pairs(self, mock_estimate: object) -> None:
        from research_agent.memory import WorkingMemory

        mock_estimate.side_effect = lambda model, messages: len(messages) * 10

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=30)
        memory.add_user("What is the weather?")
        memory.add_assistant(
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                }
            ]
        )
        memory.add_tool_result("call_123", "Sunny, 72F")
        memory.add_user("Thanks!")
        memory.add_assistant("You're welcome!")

        messages = memory.to_list(truncate=True)

        tool_call_idx = next((i for i, m in enumerate(messages) if m.get("tool_calls")), None)
        if tool_call_idx is not None:
            tool_result_idx = next(
                (i for i, m in enumerate(messages) if m.get("tool_call_id") == "call_123"),
                None,
            )
            assert tool_result_idx is not None
            assert tool_result_idx == tool_call_idx + 1

    @patch("research_agent.memory.working.estimate_tokens")
    def test_cutoff_on_tool_message_finds_assistant(self, mock_estimate: object) -> None:
        from research_agent.memory import WorkingMemory

        mock_estimate.side_effect = lambda model, messages: len(messages) * 10

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=25)
        memory.add_user("Query 1")
        memory.add_assistant(
            tool_calls=[
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ]
        )
        memory.add_tool_result("call_abc", "Result 1")
        memory.add_user("Query 2")

        messages = memory.to_list(truncate=True)

        has_tool_calls = any(m.get("tool_calls") for m in messages)
        has_tool_result = any(m.get("tool_call_id") == "call_abc" for m in messages)

        assert has_tool_calls == has_tool_result

    @patch("research_agent.memory.working.estimate_tokens")
    def test_cutoff_with_multiple_tool_results_finds_shared_assistant(
        self, mock_estimate: object
    ) -> None:
        from research_agent.memory import WorkingMemory

        mock_estimate.side_effect = lambda model, messages: len(messages) * 10

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=30)
        memory.add_user("Get weather for multiple cities")
        memory.add_assistant(
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                },
            ]
        )
        memory.add_tool_result("call_1", "Paris: Sunny")
        memory.add_tool_result("call_2", "London: Rainy")
        memory.add_user("Thanks!")

        messages = memory.to_list(truncate=True)

        tool_call_ids = set()
        for m in messages:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    tool_call_ids.add(tc["id"])
            if m.get("tool_call_id"):
                assert m["tool_call_id"] in tool_call_ids

    def test_orphan_tool_message_handled_gracefully(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=1000)
        memory.add_tool_result("call_orphan", "Result without preceding call")

        messages = memory.to_list(truncate=False)
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"

    @patch("research_agent.memory.working.estimate_tokens")
    def test_parallel_tool_calls_preserved_together(self, mock_estimate: object) -> None:
        from research_agent.memory import WorkingMemory

        mock_estimate.side_effect = lambda model, messages: len(messages) * 10

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=40)
        memory.add_user("Search for both")
        memory.add_assistant(
            tool_calls=[
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                },
            ]
        )
        memory.add_tool_result("call_a", "Result A")
        memory.add_tool_result("call_b", "Result B")

        messages = memory.to_list(truncate=True)

        tool_call_ids_included = set()
        for m in messages:
            if tc := m.get("tool_calls"):
                for t in tc:
                    tool_call_ids_included.add(t["id"])

        tool_result_ids_included = set()
        for m in messages:
            if tcid := m.get("tool_call_id"):
                tool_result_ids_included.add(tcid)

        if tool_call_ids_included:
            assert tool_result_ids_included.issubset(tool_call_ids_included)


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
    def test_default_backend_is_sliding_window(self) -> None:
        from research_agent.memory import WorkingMemory
        from research_agent.memory.working import SlidingWindowMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        assert isinstance(memory._backend, SlidingWindowMemory)

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

            def get_for_context(self, max_tokens: int, token_counter: object) -> list[dict]:
                return self.get_all()

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

            def get_for_context(self, max_tokens: int, token_counter: object) -> list[dict]:
                return self.get_all()

            def clear(self) -> None:
                self.messages = []

        backend = TrackingBackend()
        memory = WorkingMemory(model="gpt-4o-mini", backend=backend)
        memory.add_user("Test")

        assert backend.add_called


class TestEdgeCases:
    @patch("research_agent.memory.working.estimate_tokens")
    def test_all_messages_exceed_context_returns_minimum(self, mock_estimate: object) -> None:
        from research_agent.memory import WorkingMemory

        mock_estimate.side_effect = lambda model, messages: len(messages) * 10

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=5)
        for i in range(10):
            memory.add_user(f"Message number {i}")

        messages = memory.to_list(truncate=True)
        assert len(messages) >= 1

    @patch("research_agent.memory.working.estimate_tokens")
    def test_message_exactly_at_token_limit(self, mock_estimate: object) -> None:
        from research_agent.memory import WorkingMemory

        mock_estimate.side_effect = lambda model, messages: len(messages) * 10

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=10)
        memory.add_user("Hi")

        messages = memory.to_list(truncate=True)
        assert len(messages) == 1

    def test_empty_memory_returns_empty_list(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        messages = memory.to_list(truncate=False)
        assert messages == []

    def test_empty_memory_with_system_returns_system(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.set_system("System prompt")

        messages = memory.to_list(truncate=False)
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_single_message_fits_within_context(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini", max_tokens=1000)
        memory.add_user("Hello")

        messages = memory.to_list(truncate=True)
        assert len(messages) == 1

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

        messages = memory.to_list(truncate=False)
        assert messages[0]["content"] == "Let me search for that."
        assert messages[0]["tool_calls"] is not None

    def test_tool_result_with_error_content(self) -> None:
        from research_agent.memory import WorkingMemory

        memory = WorkingMemory(model="gpt-4o-mini")
        memory.add_tool_result("call_123", "Error: API timeout")

        messages = memory.to_list(truncate=False)
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
