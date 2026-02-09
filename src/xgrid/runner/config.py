from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, cast

from .grid import BoundVariableSpec


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in config: {path}") from exc


def validate_variable_entries(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variables = config.get("variables")
    if not isinstance(variables, dict):
        raise SystemExit("Config must contain a 'variables' object")

    typed_vars: dict[str, dict[str, Any]] = {}
    for variable_key, entry in variables.items():
        if not isinstance(entry, dict):
            raise SystemExit(f"Config for variable '{variable_key}' must be an object")
        generator = entry.get("generator")
        if not isinstance(generator, str) or not generator.strip():
            raise SystemExit(
                f"Config must define non-empty string 'variables.{variable_key}.generator'"
            )
        typed_vars[variable_key] = dict(entry)
    return typed_vars


def validate_experiment_entries(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    experiments = config.get("experiments")
    if not isinstance(experiments, dict):
        raise SystemExit("Config must contain an 'experiments' object")
    if not experiments:
        raise SystemExit(
            "Config must define at least one experiment under 'experiments'"
        )

    typed_experiments: dict[str, dict[str, Any]] = {}
    for experiment_key, entry in experiments.items():
        if not isinstance(entry, dict):
            raise SystemExit(
                f"Config for experiment '{experiment_key}' must be an object"
            )
        fn_name = entry.get("fn")
        if not isinstance(fn_name, str) or not fn_name.strip():
            raise SystemExit(
                f"Config must define non-empty string 'experiments.{experiment_key}.fn'"
            )
        bindings = entry.get("bindings")
        if not isinstance(bindings, dict):
            raise SystemExit(
                f"Config must define object 'experiments.{experiment_key}.bindings'"
            )
        typed_experiments[experiment_key] = {
            "fn": fn_name.strip(),
            "bindings": dict(bindings),
        }
    return typed_experiments


def warn_unused_variables(
    *,
    variables: dict[str, dict[str, Any]],
    experiments: dict[str, dict[str, Any]],
) -> None:
    referenced: set[str] = set()
    for entry in experiments.values():
        bindings = entry.get("bindings")
        if not isinstance(bindings, dict):
            continue
        for value in bindings.values():
            if isinstance(value, str) and value.strip():
                referenced.add(value)

    extra = [key for key in variables.keys() if key not in referenced]
    if extra:
        warnings.warn(f"Unknown variables in config: {', '.join(extra)}", stacklevel=2)


def resolve_bound_variables(
    fn: Callable[..., Any],
    *,
    experiment_key: str,
    bindings: dict[str, Any],
    variables: dict[str, dict[str, Any]],
    module: object,
) -> tuple[list[BoundVariableSpec], dict[str, dict[str, Any]]]:
    typed_bindings = validate_experiment_bindings(
        fn,
        experiment_key=experiment_key,
        bindings=bindings,
    )

    bound: list[BoundVariableSpec] = []
    config_vars: dict[str, dict[str, Any]] = {}
    for argument_name, variable_key in typed_bindings.items():
        variable_config = variables.get(variable_key)
        if variable_config is None:
            raise SystemExit(
                f"Unknown variable '{variable_key}' bound to argument '{argument_name}' "
                f"for experiment '{experiment_key}'"
            )
        generator_name = variable_config.get("generator")
        if not isinstance(generator_name, str) or not generator_name.strip():
            raise SystemExit(
                f"Config must define non-empty string 'variables.{variable_key}.generator'"
            )
        generator = getattr(module, generator_name, None)
        if not callable(generator):
            raise SystemExit(
                f"Unknown generator '{generator_name}' for variable '{variable_key}' "
                f"bound to argument '{argument_name}' for experiment '{experiment_key}'"
            )
        generator_typed = cast(
            Callable[..., Iterable[tuple[object, Mapping[str, object]]]], generator
        )
        config_vars[variable_key] = dict(variable_config)
        bound.append(
            BoundVariableSpec(
                argument_name=argument_name,
                variable_key=variable_key,
                generator_name=generator_name,
                generator=generator_typed,
            )
        )
    return bound, config_vars


def validate_experiment_bindings(
    fn: Callable[..., Any],
    *,
    experiment_key: str,
    bindings: dict[str, Any],
) -> dict[str, str]:
    signature = inspect.signature(fn)
    parameters = list(signature.parameters.values())
    bindable_parameters: list[str] = []
    required_parameters: list[str] = []
    for parameter in parameters:
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            bindable_parameters.append(parameter.name)
            if parameter.default is inspect.Parameter.empty:
                required_parameters.append(parameter.name)

    unknown = [key for key in bindings.keys() if key not in bindable_parameters]
    if unknown:
        raise SystemExit(
            f"Unknown bindings for experiment '{experiment_key}': {', '.join(unknown)}"
        )

    missing_required = [name for name in required_parameters if name not in bindings]
    if missing_required:
        raise SystemExit(
            f"Missing bindings for experiment '{experiment_key}': "
            f"{', '.join(missing_required)}"
        )

    typed_bindings: dict[str, str] = {}
    for parameter_name in bindable_parameters:
        if parameter_name not in bindings:
            continue
        variable_name = bindings[parameter_name]
        if not isinstance(variable_name, str) or not variable_name.strip():
            raise SystemExit(
                f"Binding for argument '{parameter_name}' in experiment "
                f"'{experiment_key}' must be a non-empty string"
            )
        typed_bindings[parameter_name] = variable_name
    return typed_bindings
