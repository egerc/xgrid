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
    variables: list[str] | None


_VARIABLES: dict[str, VariableSpec[object, object]] = {}
_EXPERIMENTS: dict[str, ExperimentSpec] = {}


def variable(
    *, name: str
) -> Callable[
    [VariableGenerator[P, ValueT, MetaT]], VariableGenerator[P, ValueT, MetaT]
]:
    """Return a decorator that registers a variable generator by name.

    Args:
        name: Registry key to associate with the decorated variable generator.

    Returns:
        A decorator that stores the generator in the variable registry and
        returns the original function unchanged.

    Raises:
        ValueError: If a variable with the same name is already registered.
    """

    def decorator(
        fn: VariableGenerator[P, ValueT, MetaT],
    ) -> VariableGenerator[P, ValueT, MetaT]:
        if name in _VARIABLES:
            raise ValueError(f"Variable '{name}' is already registered")
        _VARIABLES[name] = VariableSpec(name=name, generator=fn)
        return fn

    return decorator


def experiment(
    *, variables: list[str] | None = None
) -> Callable[[Callable[P, ResultT]], Callable[P, ResultT | ExperimentRows]]:
    """Return a decorator that registers an experiment function.

    Args:
        variables: Optional list of variable names that the experiment should
            run against. When omitted, all registered variables are available.

    Returns:
        A decorator that registers the experiment metadata and wraps the
        original callable while preserving its signature.

    Raises:
        ValueError: If an experiment with the same function name exists.
    """

    def decorator(fn: Callable[P, ResultT]) -> Callable[P, ResultT | ExperimentRows]:
        if fn.__name__ in _EXPERIMENTS:
            raise ValueError(f"Experiment '{fn.__name__}' is already registered")
        _EXPERIMENTS[fn.__name__] = ExperimentSpec(
            name=fn.__name__, fn=fn, variables=variables
        )

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> ResultT | ExperimentRows:
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def get_variable_registry() -> dict[str, VariableSpec[object, object]]:
    return _VARIABLES


def get_experiment_registry() -> dict[str, ExperimentSpec]:
    return _EXPERIMENTS


def _clear_registry() -> None:
    _VARIABLES.clear()
    _EXPERIMENTS.clear()
