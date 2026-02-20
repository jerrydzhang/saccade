import warnings

import pytest

pytestmark = pytest.mark.unit


class TestToolDecoratorName:
    def test_extracts_name_from_function_name(self):
        from research_agent.tools import tool

        @tool
        def web_search(query: str): ...

        assert web_search.name == "web_search"

    def test_extracts_name_from_camel_case(self):
        from research_agent.tools import tool

        @tool
        def webSearch(query: str): ...

        assert webSearch.name == "webSearch"


class TestToolDecoratorDescription:
    def test_extracts_description_from_docstring(self):
        from research_agent.tools import tool

        @tool
        def search(query: str):
            """Search the web for information."""
            ...

        assert search.description == "Search the web for information."

    def test_uses_empty_string_if_no_docstring(self):
        from research_agent.tools import tool

        @tool
        def no_docs(query: str): ...

        assert no_docs.description == ""

    def test_multiline_docstring_uses_first_line(self):
        from research_agent.tools import tool

        @tool
        def search(query: str):
            """Search the web.

            Additional details here.
            More info.
            """
            ...

        assert search.description == "Search the web."


class TestToolSchemaBasicTypes:
    def test_schema_for_str_type(self):
        from research_agent.tools import tool

        @tool
        def echo(text: str): ...

        schema = echo.schema
        assert schema["properties"]["text"]["type"] == "string"
        assert "text" in schema["required"]

    def test_schema_for_int_type(self):
        from research_agent.tools import tool

        @tool
        def count_items(limit: int): ...

        schema = count_items.schema
        assert schema["properties"]["limit"]["type"] == "integer"
        assert "limit" in schema["required"]

    def test_schema_for_float_type(self):
        from research_agent.tools import tool

        @tool
        def calculate(price: float): ...

        schema = calculate.schema
        assert schema["properties"]["price"]["type"] == "number"
        assert "price" in schema["required"]

    def test_schema_for_bool_type(self):
        from research_agent.tools import tool

        @tool
        def toggle(enabled: bool): ...

        schema = toggle.schema
        assert schema["properties"]["enabled"]["type"] == "boolean"
        assert "enabled" in schema["required"]


class TestToolSchemaOptionalParams:
    def test_optional_param_not_in_required(self):
        from research_agent.tools import tool

        @tool
        def search(query: str, max_results: int = 5): ...

        schema = search.schema
        assert "query" in schema["required"]
        assert "max_results" not in schema.get("required", [])

    def test_optional_param_has_type(self):
        from research_agent.tools import tool

        @tool
        def search(query: str, max_results: int = 5): ...

        schema = search.schema
        assert schema["properties"]["max_results"]["type"] == "integer"


class TestToolSchemaNoParams:
    def test_schema_for_no_params(self):
        from research_agent.tools import tool

        @tool
        def ping(): ...

        schema = ping.schema
        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert schema.get("required", []) == []


class TestToolSchemaComplexTypes:
    def test_schema_for_list_of_str(self):
        from research_agent.tools import tool

        @tool
        def process_items(items: list[str]): ...

        schema = process_items.schema
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "string"

    def test_schema_for_dict_str_int(self):
        from research_agent.tools import tool

        @tool
        def process_scores(scores: dict[str, int]): ...

        schema = process_scores.schema
        assert schema["properties"]["scores"]["type"] == "object"

    def test_schema_for_literal_enum(self):
        from typing import Literal

        from research_agent.tools import tool

        @tool
        def set_mode(mode: Literal["fast", "thorough"]): ...

        schema = set_mode.schema
        assert schema["properties"]["mode"]["enum"] == ["fast", "thorough"]


class TestToolSchemaRequiredVsOptional:
    def test_all_params_required_when_no_defaults(self):
        from research_agent.tools import tool

        @tool
        def fetch(url: str, timeout: int): ...

        schema = fetch.schema
        assert set(schema["required"]) == {"url", "timeout"}

    def test_mix_of_required_and_optional(self):
        from research_agent.tools import tool

        @tool
        def search(
            query: str,
            max_results: int = 10,
            include_snippets: bool = True,
        ): ...

        schema = search.schema
        assert schema["required"] == ["query"]


class TestToolSchemaOpenAIFormat:
    def test_schema_has_type_properties_required(self):
        from research_agent.tools import tool

        @tool
        def search(query: str): ...

        schema = search.schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_openai_function_format(self):
        from research_agent.tools import tool

        @tool
        def search(query: str): ...

        func_schema = search.to_openai_schema()
        assert func_schema["type"] == "function"
        assert func_schema["function"]["name"] == "search"
        assert func_schema["function"]["description"] == ""
        assert "parameters" in func_schema["function"]

    def test_openai_schema_includes_description_from_docstring(self):
        from research_agent.tools import tool

        @tool
        def search(query: str):
            """Search the web for information."""
            ...

        func_schema = search.to_openai_schema()
        assert (
            func_schema["function"]["description"] == "Search the web for information."
        )


class TestDirectToolCalls:
    def test_sync_tool_callable_without_await(self):
        from research_agent.tools import tool

        @tool
        def greet(name: str):
            return {"greeting": f"Hello, {name}!"}

        result = greet("World")
        assert result == {"greeting": "Hello, World!"}

    @pytest.mark.asyncio
    async def test_async_tool_requires_await(self):
        from research_agent.tools import tool

        @tool
        async def fetch(url: str):
            return {"url": url}

        result = await fetch("http://example.com")
        assert result == {"url": "http://example.com"}

    def test_sync_direct_call_traces(self):
        from saccade import Trace

        from research_agent.tools import tool

        @tool
        def my_tool(value: str):
            return value.upper()

        with Trace() as trace:
            my_tool("test")

        assert len(trace.events) > 0

    def test_sync_direct_call_raises_on_error(self):
        from research_agent.tools import tool

        @tool
        def failing_tool(value: str):
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            failing_tool("test")

    def test_sync_direct_call_no_type_coercion(self):
        from research_agent.tools import tool

        @tool
        def count(limit: int):
            return {"limit": limit}

        with pytest.raises(TypeError):
            count("not_an_int")

    @pytest.mark.asyncio
    async def test_async_direct_call_traces(self):
        from saccade import Trace

        from research_agent.tools import tool

        @tool
        async def async_tool(value: str):
            return value.upper()

        with Trace() as trace:
            await async_tool("test")

        assert len(trace.events) > 0


class TestExecuteMethod:
    @pytest.mark.asyncio
    async def test_execute_with_dict_input(self):
        from research_agent.tools import tool

        executed = []

        @tool
        def my_tool(value: int):
            executed.append(value)
            return {"value": value}

        result = await my_tool.execute({"value": "42"})
        assert result == {"value": 42}
        assert executed == [42]

    @pytest.mark.asyncio
    async def test_execute_coerces_types(self):
        from research_agent.tools import tool

        @tool
        def count(limit: int):
            return {"limit": limit}

        result = await count.execute({"limit": "5"})
        assert result == {"limit": 5}

    @pytest.mark.asyncio
    async def test_execute_validation_failure_returns_tool_error(self):
        from research_agent.tools import tool, ToolError

        @tool
        def count(limit: int):
            return {"limit": limit}

        result = await count.execute({"limit": "abc"})
        assert isinstance(result, ToolError)
        assert "limit" in result.error
        assert "int" in result.error.lower() or "integer" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_traces(self):
        from saccade import Trace

        from research_agent.tools import tool

        @tool
        def my_tool(value: str):
            return value

        with Trace() as trace:
            await my_tool.execute({"value": "test"})

        assert len(trace.events) > 0

    @pytest.mark.asyncio
    async def test_execute_missing_required_returns_tool_error(self):
        from research_agent.tools import tool, ToolError

        @tool
        def search(query: str, max_results: int):
            return {"query": query}

        result = await search.execute({"query": "test"})
        assert isinstance(result, ToolError)
        assert "max_results" in result.error
        assert "required" in result.error.lower() or "missing" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_coerces_bool_true(self):
        from research_agent.tools import tool

        @tool
        def toggle(enabled: bool):
            return {"enabled": enabled}

        result = await toggle.execute({"enabled": "true"})
        assert result == {"enabled": True}

    @pytest.mark.asyncio
    async def test_execute_coerces_bool_false(self):
        from research_agent.tools import tool

        @tool
        def toggle(enabled: bool):
            return {"enabled": enabled}

        result = await toggle.execute({"enabled": "false"})
        assert result == {"enabled": False}

    @pytest.mark.asyncio
    async def test_str_return_distinguished_from_error(self):
        from research_agent.tools import tool, ToolError

        @tool
        def count(limit: int):
            return f"counting to {limit}"

        success = await count.execute({"limit": 10})
        assert not isinstance(success, ToolError)
        assert success == "counting to 10"

        fail = await count.execute({"limit": "not_a_number"})
        assert isinstance(fail, ToolError)

    @pytest.mark.asyncio
    async def test_execute_ignores_unknown_parameters(self):
        from research_agent.tools import tool

        @tool
        def greet(name: str):
            return f"Hello, {name}!"

        result = await greet.execute({"name": "World", "unknown": "ignored"})
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        from research_agent.tools import tool

        @tool
        async def async_fetch(url: str):
            return {"url": url, "fetched": True}

        result = await async_fetch.execute({"url": "http://example.com"})
        assert result == {"url": "http://example.com", "fetched": True}

    @pytest.mark.asyncio
    async def test_execute_async_tool_error_returns_tool_error(self):
        from research_agent.tools import tool, ToolError

        @tool
        async def async_failing(value: str):
            raise ValueError("async boom")

        result = await async_failing.execute({"value": "test"})
        assert isinstance(result, ToolError)
        assert "boom" in result.error


class TestToolTracing:
    def test_span_has_kind_tool_and_name(self):
        from saccade import Trace, project_tree

        from research_agent.tools import tool

        @tool
        def my_special_tool(query: str):
            return query

        with Trace() as trace:
            my_special_tool("test")

        tree = project_tree(trace.events)
        assert tree.roots[0].name == "my_special_tool"
        assert tree.roots[0].kind == "tool"

    def test_span_captures_inputs(self):
        from saccade import Trace

        from research_agent.tools import tool

        @tool
        def search(query: str, limit: int = 10):
            return {"query": query}

        with Trace() as trace:
            search("test", limit=5)

        start_event = next(e for e in trace.events if e.type.name == "START")
        assert start_event.inputs is not None
        assert start_event.inputs.get("query") == "test"
        assert start_event.inputs.get("limit") == 5

    def test_span_captures_output(self):
        from saccade import Trace

        from research_agent.tools import tool

        @tool
        def greet(name: str):
            return {"greeting": f"Hello, {name}!"}

        with Trace() as trace:
            greet("World")

        output_event = next(e for e in trace.events if e.type.name == "OUTPUT")
        assert output_event.output == {"greeting": "Hello, World!"}

    def test_span_captures_error_on_failure(self):
        from saccade import Trace

        from research_agent.tools import tool

        @tool
        def failing_tool(value: str):
            raise ValueError("boom")

        with Trace() as trace:
            with pytest.raises(ValueError):
                failing_tool("test")

        error_event = next((e for e in trace.events if e.type.name == "ERROR"), None)
        assert error_event is not None
        assert "boom" in error_event.error


class TestToolErrorHandling:
    @pytest.mark.asyncio
    async def test_exception_becomes_tool_error(self):
        from research_agent.tools import tool, ToolError

        @tool
        def failing_tool(value: str):
            raise ValueError("Connection timeout")

        result = await failing_tool.execute({"value": "test"})
        assert isinstance(result, ToolError)
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_tool_error_exception_for_custom_message(self):
        from research_agent.tools import tool, ToolError

        @tool
        def custom_error_tool(value: str):
            raise ToolError("Custom error: invalid API key")

        result = await custom_error_tool.execute({"value": "test"})
        assert isinstance(result, ToolError)
        assert "Custom error: invalid API key" in result.error

    @pytest.mark.asyncio
    async def test_validation_error_includes_field_name_and_type(self):
        from research_agent.tools import tool, ToolError

        @tool
        def count(limit: int):
            return {"limit": limit}

        result = await count.execute({"limit": "abc"})
        assert isinstance(result, ToolError)
        assert "limit" in result.error
        assert "int" in result.error.lower()

    @pytest.mark.asyncio
    async def test_validation_error_lists_all_missing_fields(self):
        from research_agent.tools import tool, ToolError

        @tool
        def strict_tool(count: int, name: str):
            return {"name": name, "count": count}

        result = await strict_tool.execute({})
        assert isinstance(result, ToolError)
        assert "count" in result.error and "name" in result.error


class TestToolRegistry:
    def test_register_adds_tool(self):
        from research_agent.tools import tool, ToolRegistry

        @tool
        def my_tool(value: str):
            return value

        registry = ToolRegistry()
        registry.register(my_tool)

        assert "my_tool" in registry

    def test_get_schemas_returns_all(self):
        from research_agent.tools import tool, ToolRegistry

        @tool
        def tool_a(value: str):
            return value

        @tool
        def tool_b(count: int):
            return {"count": count}

        registry = ToolRegistry()
        registry.register(tool_a)
        registry.register(tool_b)

        schemas = registry.get_schemas()
        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"tool_a", "tool_b"}

    def test_get_schema_single(self):
        from research_agent.tools import tool, ToolRegistry

        @tool
        def my_tool(value: str):
            return value

        registry = ToolRegistry()
        registry.register(my_tool)

        schema = registry.get_schema("my_tool")
        assert schema["function"]["name"] == "my_tool"

    def test_duplicate_registration_raises_error(self):
        from research_agent.tools import tool, ToolRegistry

        @tool
        def my_tool(value: str):
            return value

        registry = ToolRegistry()
        registry.register(my_tool)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(my_tool)

    @pytest.mark.asyncio
    async def test_execute_nonexistent_returns_tool_error(self):
        from research_agent.tools import ToolError, ToolRegistry

        registry = ToolRegistry()
        result = await registry.execute("nonexistent", {"value": "test"})
        assert isinstance(result, ToolError)
        assert "not found" in result.error.lower()

    def test_from_functions_convenience(self):
        from research_agent.tools import tool, ToolRegistry

        @tool
        def tool_a(value: str):
            return value

        @tool
        def tool_b(count: int):
            return {"count": count}

        registry = ToolRegistry.from_functions([tool_a, tool_b])

        assert "tool_a" in registry
        assert "tool_b" in registry

    def test_from_functions_rejects_undecorated(self):
        from research_agent.tools import ToolRegistry

        def plain_func(value: str):
            return value

        with pytest.raises(TypeError, match="ToolDefinition|decorated"):
            ToolRegistry.from_functions([plain_func])

    def test_register_rejects_undecorated(self):
        from research_agent.tools import ToolRegistry

        def plain_func(value: str):
            return value

        registry = ToolRegistry()
        with pytest.raises(TypeError, match="ToolDefinition|decorated"):
            registry.register(plain_func)


class TestToolIntegration:
    @pytest.mark.asyncio
    async def test_tool_schema_openai_format(self):
        from research_agent.tools import tool, ToolRegistry

        @tool
        def web_search(query: str, max_results: int = 5):
            return {"results": [], "query": query}

        registry = ToolRegistry.from_functions([web_search])
        schemas = registry.get_schemas()

        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "web_search"
        assert schema["function"]["description"] == ""
        assert "parameters" in schema["function"]

    @pytest.mark.asyncio
    async def test_sequential_tool_calls_create_sibling_spans(self):
        from saccade import Trace, project_tree

        from research_agent.tools import tool

        @tool
        def outer_tool(value: str):
            return f"outer({value})"

        @tool
        def inner_tool(value: str):
            return f"inner({value})"

        with Trace() as trace:
            await outer_tool.execute({"value": "test"})
            await inner_tool.execute({"value": "nested"})

        tree = project_tree(trace.events)
        assert len(tree.roots) == 2


class TestToolReturnTypes:
    @pytest.mark.asyncio
    async def test_returns_string(self):
        from research_agent.tools import tool

        @tool
        def echo(value: str):
            return f"echo: {value}"

        result = await echo.execute({"value": "hello"})
        assert result == "echo: hello"

    @pytest.mark.asyncio
    async def test_returns_dict(self):
        from research_agent.tools import tool

        @tool
        def get_user(user_id: str):
            return {"id": user_id, "name": "Test User", "active": True}

        result = await get_user.execute({"user_id": "123"})
        assert result == {"id": "123", "name": "Test User", "active": True}

    @pytest.mark.asyncio
    async def test_returns_list(self):
        from research_agent.tools import tool

        @tool
        def list_items(limit: int):
            return [{"id": i, "name": f"item_{i}"} for i in range(limit)]

        result = await list_items.execute({"limit": 3})
        assert result == [
            {"id": 0, "name": "item_0"},
            {"id": 1, "name": "item_1"},
            {"id": 2, "name": "item_2"},
        ]

    @pytest.mark.asyncio
    async def test_returns_int(self):
        from research_agent.tools import tool

        @tool
        def count_items():
            return 42

        result = await count_items.execute({})
        assert result == 42

    @pytest.mark.asyncio
    async def test_returns_none(self):
        from research_agent.tools import tool

        @tool
        def do_nothing():
            return None

        result = await do_nothing.execute({})
        assert result is None


class TestToolEdgeCases:
    @pytest.mark.asyncio
    async def test_no_active_span_warns_but_continues(self):
        from research_agent.tools import tool

        @tool
        def simple_tool(value: str):
            return value

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await simple_tool.execute({"value": "test"})

            assert result == "test"
            assert len(caught) == 1
            assert "no active span" in str(caught[0].message).lower()

    @pytest.mark.asyncio
    async def test_tool_with_no_params(self):
        from research_agent.tools import tool

        @tool
        def no_params():
            return "ok"

        result = await no_params.execute({})
        assert result == "ok"

    def test_rejects_varargs(self):
        from research_agent.tools import tool

        with pytest.raises(ValueError, match="\\*args|variadic"):

            @tool
            def bad_tool(*args: str):
                return "ok"

    def test_rejects_kwargs(self):
        from research_agent.tools import tool

        with pytest.raises(ValueError, match="\\*\\*kwargs|keyword"):

            @tool
            def bad_tool(**kwargs: str):
                return "ok"

    def test_missing_type_annotation_defaults_to_str(self):
        from research_agent.tools import tool

        @tool
        def flexible(value="default"):
            return value

        schema = flexible.schema
        assert schema["properties"]["value"]["type"] == "string"


class TestToolSchemaOptionalTypes:
    def test_optional_str_schema(self):
        from typing import Optional

        from research_agent.tools import tool

        @tool
        def search(query: str, filter: Optional[str] = None):
            return {"query": query}

        schema = search.schema
        assert schema["properties"]["filter"]["type"] == "string"
        assert "filter" not in schema.get("required", [])

    def test_optional_int_schema(self):
        from typing import Optional

        from research_agent.tools import tool

        @tool
        def fetch(url: str, timeout: Optional[int] = None):
            return {"url": url}

        schema = fetch.schema
        assert schema["properties"]["timeout"]["type"] == "integer"
        assert "timeout" not in schema.get("required", [])

    def test_pipe_none_syntax(self):
        from research_agent.tools import tool

        @tool
        def search(query: str, filter: str | None = None):
            return {"query": query}

        schema = search.schema
        assert schema["properties"]["filter"]["type"] == "string"
        assert "filter" not in schema.get("required", [])

    @pytest.mark.asyncio
    async def test_execute_accepts_none_for_optional(self):
        from research_agent.tools import tool

        @tool
        def search(query: str, filter: str | None = None):
            return {"query": query, "filter": filter}

        result = await search.execute({"query": "test", "filter": None})
        assert result == {"query": "test", "filter": None}

    @pytest.mark.asyncio
    async def test_execute_omitted_optional_gets_none(self):
        from research_agent.tools import tool

        @tool
        def search(query: str, filter: str | None = None):
            return {"query": query, "filter": filter}

        result = await search.execute({"query": "test"})
        assert result == {"query": "test", "filter": None}


class TestToolSchemaUnionTypes:
    def test_union_str_int_schema(self):
        from typing import Union

        from research_agent.tools import tool

        @tool
        def process(value: Union[str, int]):
            return {"value": value}

        schema = process.schema
        assert "anyOf" in schema["properties"]["value"]
        types = {t["type"] for t in schema["properties"]["value"]["anyOf"]}
        assert types == {"string", "integer"}

    def test_pipe_union_syntax(self):
        from research_agent.tools import tool

        @tool
        def process(value: str | int):
            return {"value": value}

        schema = process.schema
        assert "anyOf" in schema["properties"]["value"]
        types = {t["type"] for t in schema["properties"]["value"]["anyOf"]}
        assert types == {"string", "integer"}

    @pytest.mark.asyncio
    async def test_execute_union_accepts_string(self):
        from research_agent.tools import tool

        @tool
        def process(value: str | int):
            return {"value": value, "type": type(value).__name__}

        result = await process.execute({"value": "hello"})
        assert result == {"value": "hello", "type": "str"}

    @pytest.mark.asyncio
    async def test_execute_union_accepts_int(self):
        from research_agent.tools import tool

        @tool
        def process(value: str | int):
            return {"value": value, "type": type(value).__name__}

        result = await process.execute({"value": 42})
        assert result == {"value": 42, "type": "int"}

    @pytest.mark.asyncio
    async def test_execute_union_accepts_coerced_int(self):
        from research_agent.tools import tool

        @tool
        def process(value: str | int):
            return {"value": value, "type": type(value).__name__}

        result = await process.execute({"value": "42"})
        assert result == {"value": 42, "type": "int"}

    def test_union_with_none_optional(self):
        from research_agent.tools import tool

        @tool
        def process(value: str | int | None = None):
            return {"value": value}

        schema = process.schema
        assert "value" not in schema.get("required", [])


class TestToolSchemaNestedGenerics:
    def test_list_of_list_of_str(self):
        from research_agent.tools import tool

        @tool
        def process(matrix: list[list[str]]):
            return {"matrix": matrix}

        schema = process.schema
        prop = schema["properties"]["matrix"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "array"
        assert prop["items"]["items"]["type"] == "string"

    def test_dict_str_list_int(self):
        from research_agent.tools import tool

        @tool
        def process(scores: dict[str, list[int]]):
            return {"scores": scores}

        schema = process.schema
        prop = schema["properties"]["scores"]
        assert prop["type"] == "object"
        assert prop["additionalProperties"]["type"] == "array"
        assert prop["additionalProperties"]["items"]["type"] == "integer"

    def test_list_of_dict(self):
        from research_agent.tools import tool

        @tool
        def process(items: list[dict[str, str]]):
            return {"items": items}

        schema = process.schema
        prop = schema["properties"]["items"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_execute_nested_list(self):
        from research_agent.tools import tool

        @tool
        def process(matrix: list[list[str]]):
            return {"rows": len(matrix)}

        result = await process.execute({"matrix": [["a", "b"], ["c"]]})
        assert result == {"rows": 2}

    @pytest.mark.asyncio
    async def test_execute_nested_dict(self):
        from research_agent.tools import tool

        @tool
        def process(scores: dict[str, list[int]]):
            return {"keys": list(scores.keys())}

        result = await process.execute({"scores": {"a": [1, 2], "b": [3]}})
        assert result == {"keys": ["a", "b"]}


class TestToolSchemaPydanticModel:
    def test_pydantic_model_schema_expands(self):
        from pydantic import BaseModel

        from research_agent.tools import tool

        class User(BaseModel):
            name: str
            age: int

        @tool
        def process_user(user: User):
            return {"processed": True}

        schema = process_user.schema
        user_prop = schema["properties"]["user"]
        assert user_prop["type"] == "object"
        assert "name" in user_prop["properties"]
        assert "age" in user_prop["properties"]
        assert user_prop["properties"]["name"]["type"] == "string"
        assert user_prop["properties"]["age"]["type"] == "integer"

    def test_pydantic_model_required_fields(self):
        from pydantic import BaseModel

        from research_agent.tools import tool

        class User(BaseModel):
            name: str
            age: int

        @tool
        def process_user(user: User):
            return {"processed": True}

        schema = process_user.schema
        user_prop = schema["properties"]["user"]
        assert set(user_prop["required"]) == {"name", "age"}

    def test_pydantic_model_with_optional_fields(self):
        from typing import Optional

        from pydantic import BaseModel

        from research_agent.tools import tool

        class User(BaseModel):
            name: str
            nickname: Optional[str] = None

        @tool
        def process_user(user: User):
            return {"processed": True}

        schema = process_user.schema
        user_prop = schema["properties"]["user"]
        assert "name" in user_prop["required"]
        assert "nickname" not in user_prop.get("required", [])

    @pytest.mark.asyncio
    async def test_execute_pydantic_model(self):
        from pydantic import BaseModel

        from research_agent.tools import tool

        class User(BaseModel):
            name: str
            age: int

        @tool
        def process_user(user: User):
            return {"greeting": f"Hello {user.name}, age {user.age}"}

        result = await process_user.execute({"user": {"name": "Alice", "age": 30}})
        assert result == {"greeting": "Hello Alice, age 30"}

    @pytest.mark.asyncio
    async def test_execute_pydantic_model_validation_error(self):
        from pydantic import BaseModel

        from research_agent.tools import tool, ToolError

        class User(BaseModel):
            name: str
            age: int

        @tool
        def process_user(user: User):
            return {"greeting": f"Hello {user.name}"}

        result = await process_user.execute(
            {"user": {"name": "Alice", "age": "not_a_number"}}
        )
        assert isinstance(result, ToolError)
        assert "age" in result.error.lower()


class TestExecuteCoercionEdgeCases:
    @pytest.mark.asyncio
    async def test_bool_coerce_uppercase_true(self):
        from research_agent.tools import tool

        @tool
        def toggle(enabled: bool):
            return {"enabled": enabled}

        result = await toggle.execute({"enabled": "TRUE"})
        assert result == {"enabled": True}

    @pytest.mark.asyncio
    async def test_bool_coerce_mixed_case_true(self):
        from research_agent.tools import tool

        @tool
        def toggle(enabled: bool):
            return {"enabled": enabled}

        result = await toggle.execute({"enabled": "True"})
        assert result == {"enabled": True}

    @pytest.mark.asyncio
    async def test_bool_coerce_uppercase_false(self):
        from research_agent.tools import tool

        @tool
        def toggle(enabled: bool):
            return {"enabled": enabled}

        result = await toggle.execute({"enabled": "FALSE"})
        assert result == {"enabled": False}

    @pytest.mark.asyncio
    async def test_bool_coerce_one(self):
        from research_agent.tools import tool

        @tool
        def toggle(enabled: bool):
            return {"enabled": enabled}

        result = await toggle.execute({"enabled": "1"})
        assert result == {"enabled": True}

    @pytest.mark.asyncio
    async def test_bool_coerce_zero(self):
        from research_agent.tools import tool

        @tool
        def toggle(enabled: bool):
            return {"enabled": enabled}

        result = await toggle.execute({"enabled": "0"})
        assert result == {"enabled": False}

    @pytest.mark.asyncio
    async def test_bool_coerce_int_one(self):
        from research_agent.tools import tool

        @tool
        def toggle(enabled: bool):
            return {"enabled": enabled}

        result = await toggle.execute({"enabled": 1})
        assert result == {"enabled": True}

    @pytest.mark.asyncio
    async def test_bool_coerce_int_zero(self):
        from research_agent.tools import tool

        @tool
        def toggle(enabled: bool):
            return {"enabled": enabled}

        result = await toggle.execute({"enabled": 0})
        assert result == {"enabled": False}

    @pytest.mark.asyncio
    async def test_bool_empty_string_is_error(self):
        from research_agent.tools import tool, ToolError

        @tool
        def toggle(enabled: bool):
            return {"enabled": enabled}

        result = await toggle.execute({"enabled": ""})
        assert isinstance(result, ToolError)

    @pytest.mark.asyncio
    async def test_bool_invalid_string_is_error(self):
        from research_agent.tools import tool, ToolError

        @tool
        def toggle(enabled: bool):
            return {"enabled": enabled}

        result = await toggle.execute({"enabled": "maybe"})
        assert isinstance(result, ToolError)

    @pytest.mark.asyncio
    async def test_int_coerce_valid_string(self):
        from research_agent.tools import tool

        @tool
        def count(limit: int):
            return {"limit": limit}

        result = await count.execute({"limit": "42"})
        assert result == {"limit": 42}

    @pytest.mark.asyncio
    async def test_int_reject_float_string_whole(self):
        from research_agent.tools import tool, ToolError

        @tool
        def count(limit: int):
            return {"limit": limit}

        result = await count.execute({"limit": "5.0"})
        assert isinstance(result, ToolError)

    @pytest.mark.asyncio
    async def test_int_reject_float_string_fractional(self):
        from research_agent.tools import tool, ToolError

        @tool
        def count(limit: int):
            return {"limit": limit}

        result = await count.execute({"limit": "5.5"})
        assert isinstance(result, ToolError)

    @pytest.mark.asyncio
    async def test_int_reject_actual_float_whole(self):
        from research_agent.tools import tool, ToolError

        @tool
        def count(limit: int):
            return {"limit": limit}

        result = await count.execute({"limit": 5.0})
        assert isinstance(result, ToolError)

    @pytest.mark.asyncio
    async def test_int_reject_actual_float_fractional(self):
        from research_agent.tools import tool, ToolError

        @tool
        def count(limit: int):
            return {"limit": limit}

        result = await count.execute({"limit": 5.5})
        assert isinstance(result, ToolError)

    @pytest.mark.asyncio
    async def test_float_coerce_valid_string(self):
        from research_agent.tools import tool

        @tool
        def calculate(price: float):
            return {"price": price}

        result = await calculate.execute({"price": "3.14"})
        assert result == {"price": 3.14}

    @pytest.mark.asyncio
    async def test_float_coerce_int_string(self):
        from research_agent.tools import tool

        @tool
        def calculate(price: float):
            return {"price": price}

        result = await calculate.execute({"price": "3"})
        assert result == {"price": 3.0}

    @pytest.mark.asyncio
    async def test_float_coerce_actual_int(self):
        from research_agent.tools import tool

        @tool
        def calculate(price: float):
            return {"price": price}

        result = await calculate.execute({"price": 3})
        assert result == {"price": 3.0}


class TestReservedParameterNames:
    def test_parameter_named_execute(self):
        from research_agent.tools import tool

        @tool
        def my_tool(execute: str):
            return {"execute": execute}

        schema = my_tool.schema
        assert "execute" in schema["properties"]
        assert schema["properties"]["execute"]["type"] == "string"

    def test_parameter_named_name(self):
        from research_agent.tools import tool

        @tool
        def my_tool(name: str):
            return {"name": name}

        assert my_tool.name == "my_tool"
        schema = my_tool.schema
        assert "name" in schema["properties"]

    def test_parameter_named_schema(self):
        from research_agent.tools import tool

        @tool
        def my_tool(schema: str):
            return {"schema": schema}

        assert "properties" in my_tool.schema
        assert "schema" in my_tool.schema["properties"]

    @pytest.mark.asyncio
    async def test_direct_call_with_reserved_name(self):
        from research_agent.tools import tool

        @tool
        def my_tool(execute: str):
            return {"execute": execute}

        result = my_tool("test_value")
        assert result == {"execute": "test_value"}

    @pytest.mark.asyncio
    async def test_execute_with_reserved_name(self):
        from research_agent.tools import tool

        @tool
        def my_tool(execute: str):
            return {"execute": execute}

        result = await my_tool.execute({"execute": "test_value"})
        assert result == {"execute": "test_value"}


class TestSchemaImmutability:
    def test_modifying_schema_does_not_affect_original(self):
        from research_agent.tools import tool

        @tool
        def my_tool(value: str):
            return value

        schema1 = my_tool.schema
        schema1["properties"]["value"]["type"] = "modified"
        schema1["new_key"] = "injected"

        schema2 = my_tool.schema
        assert schema2["properties"]["value"]["type"] == "string"
        assert "new_key" not in schema2

    def test_modifying_properties_does_not_affect_original(self):
        from research_agent.tools import tool

        @tool
        def my_tool(value: str):
            return value

        schema1 = my_tool.schema
        schema1["properties"]["injected"] = {"type": "string"}

        schema2 = my_tool.schema
        assert "injected" not in schema2["properties"]


class TestToolRegistryContains:
    def test_contains_returns_true_for_registered(self):
        from research_agent.tools import tool, ToolRegistry

        @tool
        def my_tool(value: str):
            return value

        registry = ToolRegistry()
        registry.register(my_tool)

        assert "my_tool" in registry

    def test_contains_returns_false_for_unregistered(self):
        from research_agent.tools import ToolRegistry

        registry = ToolRegistry()
        assert "nonexistent" not in registry

    def test_contains_after_multiple_registrations(self):
        from research_agent.tools import tool, ToolRegistry

        @tool
        def tool_a(value: str):
            return value

        @tool
        def tool_b(count: int):
            return {"count": count}

        registry = ToolRegistry()
        registry.register(tool_a)

        assert "tool_a" in registry
        assert "tool_b" not in registry

        registry.register(tool_b)

        assert "tool_a" in registry
        assert "tool_b" in registry
