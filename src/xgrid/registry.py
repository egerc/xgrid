from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, Generic, ParamSpec, TypeVar


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


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    fn: Callable[..., Any]


_VARIABLES: dict[str, VariableSpec[object, object]] = {}
_EXPERIMENTS: dict[str, ExperimentSpec] = {}


def variable(
    fn: VariableGenerator[P, ValueT, MetaT],
) -> VariableGenerator[P, ValueT, MetaT]:
    """Register a variable generator by function name.

    Raises:
        ValueError: If a variable with the same function name is already
            registered.
    """
    variable_name = fn.__name__
    if variable_name in _VARIABLES:
        raise ValueError(f"Variable '{variable_name}' is already registered")
    _VARIABLES[variable_name] = VariableSpec(name=variable_name, generator=fn)
    return fn


def experiment(fn: Callable[P, ResultT]) -> Callable[P, ResultT | ExperimentRows]:
    """Register an experiment function.

    Raises:
        ValueError: If an experiment with the same function name exists.
    """
    if fn.__name__ in _EXPERIMENTS:
        raise ValueError(f"Experiment '{fn.__name__}' is already registered")
    _EXPERIMENTS[fn.__name__] = ExperimentSpec(name=fn.__name__, fn=fn)

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> ResultT | ExperimentRows:
        return fn(*args, **kwargs)

    return wrapper


def get_variable_registry() -> dict[str, VariableSpec[object, object]]:
    return _VARIABLES


def get_experiment_registry() -> dict[str, ExperimentSpec]:
    return _EXPERIMENTS


def _clear_registry() -> None:
    _VARIABLES.clear()
    _EXPERIMENTS.clear()
