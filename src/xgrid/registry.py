from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Generic, ParamSpec, TypeVar


P = ParamSpec("P")
ValueT = TypeVar("ValueT", covariant=True)
MetaT = TypeVar("MetaT", covariant=True)
ResultT = TypeVar("ResultT")

VariableGenerator = Callable[P, Iterable[tuple[ValueT, Mapping[str, MetaT]]]]
ExperimentRows = list[dict[str, object]]


@dataclass(frozen=True)
class VariableSpec(Generic[ValueT, MetaT]):
    name: str
    generator: Callable[..., Iterable[tuple[ValueT, Mapping[str, MetaT]]]]


_VARIABLES: dict[str, VariableSpec[object, object]] = {}


def variable(*, name: str) -> Callable[[VariableGenerator[P, ValueT, MetaT]], VariableGenerator[P, ValueT, MetaT]]:
    def decorator(fn: VariableGenerator[P, ValueT, MetaT]) -> VariableGenerator[P, ValueT, MetaT]:
        if name in _VARIABLES:
            raise ValueError(f"Variable '{name}' is already registered")
        _VARIABLES[name] = VariableSpec(name=name, generator=fn)
        return fn

    return decorator


def experiment(*, variables: list[str] | None = None) -> Callable[[Callable[P, ResultT]], Callable[P, ResultT | ExperimentRows]]:
    def decorator(fn: Callable[P, ResultT]) -> Callable[P, ResultT | ExperimentRows]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> ResultT | ExperimentRows:
            if args or kwargs:
                return fn(*args, **kwargs)
            from .runner import run_experiment

            return run_experiment(fn, variables=variables)

        return wrapper

    return decorator


def get_variable_registry() -> dict[str, VariableSpec[object, object]]:
    return _VARIABLES


def _clear_registry() -> None:
    _VARIABLES.clear()
