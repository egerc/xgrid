from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import wraps
from typing import Any


@dataclass(frozen=True)
class VariableSpec:
    name: str
    generator: Callable[..., Iterable[tuple[Any, dict[str, Any]]]]


_VARIABLES: dict[str, VariableSpec] = {}


def variable(*, name: str) -> Callable[[Callable[..., Iterable[tuple[Any, dict[str, Any]]]]], Callable[..., Iterable[tuple[Any, dict[str, Any]]]]]:
    def decorator(
        fn: Callable[..., Iterable[tuple[Any, dict[str, Any]]]]
    ) -> Callable[..., Iterable[tuple[Any, dict[str, Any]]]]:
        if name in _VARIABLES:
            raise ValueError(f"Variable '{name}' is already registered")
        _VARIABLES[name] = VariableSpec(name=name, generator=fn)
        return fn

    return decorator


def experiment(*, variables: list[str] | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if args or kwargs:
                return fn(*args, **kwargs)
            from .runner import run_experiment

            return run_experiment(fn, variables=variables)

        return wrapper

    return decorator


def get_variable_registry() -> dict[str, VariableSpec]:
    return _VARIABLES


def _clear_registry() -> None:
    _VARIABLES.clear()
