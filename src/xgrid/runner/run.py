from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable

from . import writers
from .config import (
    resolve_bound_variables,
    validate_experiment_entries,
    validate_variable_entries,
    warn_unused_variables,
)
from .grid import build_rows_with_stats_from_bound_variables
from .logging import configure_logging
from .output import resolve_experiment_output_path, validate_output_template
from .script import load_script_module


def run_registered_experiment(
    fn: Callable[..., Any],
    *,
    config_path: Path,
    output_path: Path,
    output_format: str | None,
    show_progress: bool | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    from .config import load_config

    config = load_config(config_path)
    experiments = validate_experiment_entries(config)
    variables = validate_variable_entries(config)

    matches = [
        key for key, entry in experiments.items() if entry.get("fn") == fn.__name__
    ]
    if not matches:
        raise SystemExit(
            f"Config does not define an experiment for function '{fn.__name__}'"
        )
    if len(matches) > 1:
        available = ", ".join(sorted(matches))
        raise SystemExit(
            f"Multiple config experiments reference function '{fn.__name__}'. "
            f"Available experiments: {available}"
        )
    experiment_key = matches[0]
    experiment_entry = experiments[experiment_key]

    module = sys.modules.get(fn.__module__)
    if module is None:
        raise SystemExit(
            f"Unable to resolve module for experiment function '{fn.__name__}'"
        )

    warn_unused_variables(variables=variables, experiments=experiments)
    bound_variable_specs, config_vars = resolve_bound_variables(
        fn,
        experiment_key=experiment_key,
        bindings=experiment_entry["bindings"],
        variables=variables,
        module=module,
    )
    rows, total_iterations = build_rows_with_stats_from_bound_variables(
        fn,
        bound_variable_specs=bound_variable_specs,
        config_vars=config_vars,
        show_progress=show_progress,
        logger=logger,
    )
    writers.write_output(rows, output_path=output_path, output_format=output_format)
    if logger is not None:
        logger.info(
            "Completed run experiment=%s iterations=%d rows_written=%d output=%s",
            experiment_key,
            total_iterations,
            len(rows),
            output_path,
        )
    return rows


def run_script(
    script_path: str | Path,
    *,
    config_path: str | Path,
    output_template: str | Path,
    output_format: str | None = None,
    show_progress: bool | None = None,
    log_level: str = "INFO",
) -> dict[str, list[dict[str, Any]]]:
    logger = configure_logging(log_level)
    script_path_obj = Path(script_path)
    config_path_obj = Path(config_path)
    output_template_obj = Path(output_template)

    from .config import load_config

    config = load_config(config_path_obj)
    experiments = validate_experiment_entries(config)
    variables = validate_variable_entries(config)
    validate_output_template(
        output_template=output_template_obj,
        experiments=experiments,
        output_format=output_format,
    )

    module = load_script_module(script_path_obj)

    warn_unused_variables(variables=variables, experiments=experiments)

    results: dict[str, list[dict[str, Any]]] = {}
    for experiment_key, experiment_entry in experiments.items():
        fn_name = experiment_entry["fn"]
        fn = getattr(module, fn_name, None)
        if not callable(fn):
            raise SystemExit(
                f"Unknown experiment function '{fn_name}' for config experiment '{experiment_key}'"
            )

        output_path = resolve_experiment_output_path(
            output_template=output_template_obj,
            experiment_key=experiment_key,
            output_format=output_format,
        )
        logger.info(
            "Starting run script=%s config=%s output=%s experiment_key=%s fn=%s",
            script_path_obj,
            config_path_obj,
            output_path,
            experiment_key,
            fn_name,
        )
        bound_variable_specs, config_vars = resolve_bound_variables(
            fn,
            experiment_key=experiment_key,
            bindings=experiment_entry["bindings"],
            variables=variables,
            module=module,
        )
        rows, total_iterations = build_rows_with_stats_from_bound_variables(
            fn,
            bound_variable_specs=bound_variable_specs,
            config_vars=config_vars,
            show_progress=show_progress,
            logger=logger,
        )
        writers.write_output(rows, output_path=output_path, output_format=output_format)
        logger.info(
            "Completed run experiment_key=%s iterations=%d rows_written=%d output=%s",
            experiment_key,
            total_iterations,
            len(rows),
            output_path,
        )
        results[experiment_key] = rows
    return results


def build_rows(
    fn: Callable[..., Any],
    *,
    config: dict[str, Any],
    experiment_key: str | None = None,
    module: object | None = None,
    show_progress: bool | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    experiments = validate_experiment_entries(config)
    variables = validate_variable_entries(config)
    warn_unused_variables(variables=variables, experiments=experiments)

    selected_key: str
    if experiment_key is not None:
        if experiment_key not in experiments:
            available = ", ".join(sorted(experiments.keys()))
            raise SystemExit(
                f"Unknown experiment '{experiment_key}'. Available experiments: {available}"
            )
        selected_key = experiment_key
        entry = experiments[selected_key]
        if entry.get("fn") != fn.__name__:
            raise SystemExit(
                f"Config experiment '{selected_key}' references function '{entry.get('fn')}', "
                f"but build_rows was given '{fn.__name__}'"
            )
    else:
        matches = [
            key for key, entry in experiments.items() if entry.get("fn") == fn.__name__
        ]
        if not matches:
            raise SystemExit(
                f"Config does not define an experiment for function '{fn.__name__}'"
            )
        if len(matches) > 1:
            available = ", ".join(sorted(matches))
            raise SystemExit(
                f"Multiple config experiments reference function '{fn.__name__}'. "
                f"Available experiments: {available}"
            )
        selected_key = matches[0]
        entry = experiments[selected_key]

    resolved_module: object
    if module is None:
        resolved_module = sys.modules.get(fn.__module__)
        if resolved_module is None:
            raise SystemExit(
                f"Unable to resolve module for experiment function '{fn.__name__}'"
            )
    else:
        resolved_module = module

    bound_variable_specs, config_vars = resolve_bound_variables(
        fn,
        experiment_key=selected_key,
        bindings=entry["bindings"],
        variables=variables,
        module=resolved_module,
    )

    rows, _iteration_count = build_rows_with_stats_from_bound_variables(
        fn,
        bound_variable_specs=bound_variable_specs,
        config_vars=config_vars,
        show_progress=show_progress,
        logger=logger,
    )
    return rows
