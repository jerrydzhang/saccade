from __future__ import annotations

import copy
import inspect
import warnings
from types import UnionType
from typing import Any, Callable, Union, get_origin, get_args

from pydantic import BaseModel, create_model, ValidationError
from saccade import Span
from saccade.primitives.trace import _current_bus


class ToolError(Exception):
    def __init__(self, error: str) -> None:
        self.error = error
        super().__init__(error)

    def __repr__(self) -> str:
        return f"ToolError(error={self.error!r})"


class ToolDefinition:
    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func
        self._name = getattr(func, "__name__", "unknown")
        self._description = self._extract_description(func)
        self._is_async = inspect.iscoroutinefunction(func)
        self._signature = inspect.signature(func)
        self._validate_signature()
        self._schema_cache: dict[str, Any] | None = None
        self._pydantic_model = self._build_pydantic_model()

    def _extract_description(self, func: Callable[..., Any]) -> str:
        doc = getattr(func, "__doc__", None)
        if not doc:
            return ""
        return doc.strip().split("\n")[0].strip()

    def _validate_signature(self) -> None:
        for param in self._signature.parameters.values():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise ValueError(
                    "*args not supported - tools must have explicit parameters"
                )
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise ValueError(
                    "**kwargs not supported - tools must have explicit parameters"
                )

    def _build_pydantic_model(self) -> type[BaseModel] | None:
        fields: dict[str, Any] = {}
        for name, param in self._signature.parameters.items():
            annotation = param.annotation
            default = param.default
            if annotation is inspect.Parameter.empty:
                annotation = str
            if default is inspect.Parameter.empty:
                fields[name] = (annotation, ...)
            else:
                fields[name] = (annotation, default)
        if not fields:
            return None
        return create_model(f"{self._name}_params", **fields)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def schema(self) -> dict[str, Any]:
        if self._schema_cache is None:
            self._schema_cache = self._build_schema()
        return copy.deepcopy(self._schema_cache)

    def _build_schema(self) -> dict[str, Any]:
        if self._pydantic_model is None:
            return {"type": "object", "properties": {}, "required": []}
        schema = self._pydantic_model.model_json_schema()
        result: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        if "properties" in schema:
            for name, prop in schema["properties"].items():
                result["properties"][name] = self._convert_schema(
                    prop, schema.get("$defs", {})
                )
        required = []
        for name, param in self._signature.parameters.items():
            if param.default is inspect.Parameter.empty:
                required.append(name)
        result["required"] = required
        return result

    def _convert_schema(
        self, schema: dict[str, Any], defs: dict[str, Any]
    ) -> dict[str, Any]:
        if "$ref" in schema:
            ref_name = schema["$ref"].split("/")[-1]
            if ref_name in defs:
                return self._convert_schema(defs[ref_name], defs)
        result: dict[str, Any] = {}
        if "anyOf" in schema:
            non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
            if len(non_null) == 1:
                return self._convert_schema(non_null[0], defs)
            result["anyOf"] = [self._convert_schema(s, defs) for s in non_null]
            return result
        if "type" in schema:
            result["type"] = schema["type"]
        if "enum" in schema:
            result["enum"] = schema["enum"]
        if schema.get("type") == "array" and "items" in schema:
            result["items"] = self._convert_schema(schema["items"], defs)
        if schema.get("type") == "object":
            if "properties" in schema:
                result["properties"] = {
                    k: self._convert_schema(v, defs)
                    for k, v in schema["properties"].items()
                }
                if "required" in schema:
                    result["required"] = schema["required"]
            if "additionalProperties" in schema:
                result["additionalProperties"] = self._convert_schema(
                    schema["additionalProperties"], defs
                )
        return result

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self._description,
                "parameters": self.schema,
            },
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._validate_direct_call_args(args, kwargs)
        if self._is_async:
            return self._async_call(*args, **kwargs)
        return self._sync_call(*args, **kwargs)

    def _validate_direct_call_args(self, args: tuple, kwargs: dict) -> None:
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            param = self._signature.parameters[name]
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                continue
            self._validate_type_strict(value, annotation, name)

    def _validate_type_strict(self, value: Any, annotation: Any, name: str) -> None:
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is Union or origin is UnionType:
            for target_type in args:
                if target_type is type(None):
                    if value is None:
                        return
                    continue
                try:
                    self._validate_type_strict(value, target_type, name)
                    return
                except TypeError:
                    continue
            type_names = " or ".join(
                t.__name__ if hasattr(t, "__name__") else str(t) for t in args
            )
            raise TypeError(
                f"'{name}' must be {type_names}, got {type(value).__name__}"
            )
        if origin is list:
            if not isinstance(value, list):
                raise TypeError(f"'{name}' must be list, got {type(value).__name__}")
            return
        if origin is dict:
            if not isinstance(value, dict):
                raise TypeError(f"'{name}' must be dict, got {type(value).__name__}")
            return
        if annotation is str:
            if not isinstance(value, str):
                raise TypeError(f"'{name}' must be str, got {type(value).__name__}")
            return
        if annotation is int:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"'{name}' must be int, got {type(value).__name__}")
            return
        if annotation is float:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"'{name}' must be float, got {type(value).__name__}")
            return
        if annotation is bool:
            if not isinstance(value, bool):
                raise TypeError(f"'{name}' must be bool, got {type(value).__name__}")
            return

    def _sync_call(self, *args: Any, **kwargs: Any) -> Any:
        bound = self._bind_args(args, kwargs)
        bus = _current_bus.get()
        if bus:
            with Span(self._name, kind="tool", inputs=bound) as span:
                result = self._func(*args, **kwargs)
                span.set_output(result)
                return result
        return self._func(*args, **kwargs)

    async def _async_call(self, *args: Any, **kwargs: Any) -> Any:
        bound = self._bind_args(args, kwargs)
        bus = _current_bus.get()
        if bus:
            with Span(self._name, kind="tool", inputs=bound) as span:
                result = await self._func(*args, **kwargs)
                span.set_output(result)
                return result
        return await self._func(*args, **kwargs)

    def _bind_args(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        bound: dict[str, Any] = {}
        params = list(self._signature.parameters.keys())
        for i, arg in enumerate(args):
            if i < len(params):
                bound[params[i]] = arg
        bound.update(kwargs)
        return bound

    async def execute(self, arguments: dict[str, Any]) -> Any:
        try:
            validated = self._validate_and_coerce(arguments)
        except (ValidationError, ValueError) as e:
            return ToolError(error=str(e))
        bus = _current_bus.get()
        if bus:
            with Span(self._name, kind="tool", inputs=validated) as span:
                result = await self._execute_validated(validated)
                if not isinstance(result, ToolError):
                    span.set_output(result)
                return result
        warnings.warn(
            "Tool executed with no active span - tracing disabled",
            UserWarning,
            stacklevel=2,
        )
        return await self._execute_validated(validated)

    def _validate_and_coerce(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._pydantic_model is None:
            return {}
        coerced = dict(arguments)
        for name, param in self._signature.parameters.items():
            if name not in coerced:
                continue
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                annotation = str
            value = coerced[name]
            coerced[name] = self._coerce_value(value, annotation, name)
        model = self._pydantic_model.model_validate(coerced)
        return {
            name: getattr(model, name) for name in self._pydantic_model.model_fields
        }

    def _coerce_value(self, value: Any, annotation: Any, name: str) -> Any:
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is Union or origin is UnionType:
            if type(None) in args and value is None:
                return None
            non_none_args = [a for a in args if a is not type(None)]
            sorted_args = self._sort_union_types(non_none_args)
            for target_type in sorted_args:
                try:
                    return self._coerce_to_type(value, target_type, name)
                except (ValueError, TypeError):
                    continue
            raise ValueError(
                f"Cannot coerce {value!r} to {annotation} for parameter '{name}'"
            )
        return self._coerce_to_type(value, annotation, name)

    def _sort_union_types(self, types: list[Any]) -> list[Any]:
        order = {int: 0, float: 1, str: 2, bool: 3}
        result = []
        others = []
        for t in types:
            if t in order:
                result.append((order[t], t))
            else:
                others.append((100, t))
        result.extend(others)
        return [t for _, t in sorted(result)]

    def _coerce_to_type(self, value: Any, target_type: Any, name: str) -> Any:
        origin = get_origin(target_type)
        if origin is not None:
            if origin is list and isinstance(value, list):
                return value
            if origin is dict and isinstance(value, dict):
                return value
            return value
        if isinstance(value, target_type) and not isinstance(value, bool | int):
            return value
        if isinstance(value, target_type) and target_type not in (bool, int):
            return value
        if target_type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, int):
                if value == 0:
                    return False
                if value == 1:
                    return True
                raise ValueError(
                    f"Cannot coerce int {value} to bool for parameter '{name}'"
                )
            if isinstance(value, str):
                lower = value.lower()
                if lower in ("true", "1"):
                    return True
                if lower in ("false", "0"):
                    return False
                raise ValueError(
                    f"Cannot coerce string {value!r} to bool for parameter '{name}'"
                )
            raise ValueError(
                f"Cannot coerce {type(value).__name__} to bool for parameter '{name}'"
            )
        if target_type is int:
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, str):
                if "." in value:
                    raise ValueError(
                        f"Cannot coerce float string {value!r} to int for parameter '{name}'"
                    )
                try:
                    return int(value)
                except ValueError:
                    raise ValueError(
                        f"Cannot coerce string {value!r} to int for parameter '{name}'"
                    )
            if isinstance(value, float):
                raise ValueError(
                    f"Cannot coerce float {value} to int for parameter '{name}'"
                )
            raise ValueError(
                f"Cannot coerce {type(value).__name__} to int for parameter '{name}'"
            )
        if target_type is float:
            if isinstance(value, float):
                return value
            if isinstance(value, int) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    raise ValueError(
                        f"Cannot coerce string {value!r} to float for parameter '{name}'"
                    )
            raise ValueError(
                f"Cannot coerce {type(value).__name__} to float for parameter '{name}'"
            )
        if target_type is str:
            if isinstance(value, str):
                return value
            return str(value)
        if isinstance(target_type, type) and issubclass(target_type, BaseModel):
            if isinstance(value, target_type):
                return value
            if isinstance(value, dict):
                return target_type.model_validate(value)
            raise ValueError(
                f"Cannot coerce {type(value).__name__} to {target_type.__name__}"
            )
        return value

    async def _execute_validated(self, validated: dict[str, Any]) -> Any:
        try:
            if self._is_async:
                return await self._func(**validated)
            return self._func(**validated)
        except ToolError as e:
            return e
        except Exception as e:
            return ToolError(error=str(e))


def tool(func: Callable[..., Any]) -> ToolDefinition:
    return ToolDefinition(func)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool_def: ToolDefinition) -> None:
        if not isinstance(tool_def, ToolDefinition):
            raise TypeError(
                f"Expected ToolDefinition, got {type(tool_def).__name__}. "
                "Did you forget to decorate with @tool?"
            )
        if tool_def.name in self._tools:
            raise ValueError(f"Tool '{tool_def.name}' is already registered")
        self._tools[tool_def.name] = tool_def

    def get_schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def get_schema(self, name: str) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name].to_openai_schema()

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools:
            return ToolError(error=f"Tool '{name}' not found")
        return await self._tools[name].execute(arguments)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @classmethod
    def from_functions(cls, tools: list[ToolDefinition]) -> ToolRegistry:
        registry = cls()
        for t in tools:
            registry.register(t)
        return registry
