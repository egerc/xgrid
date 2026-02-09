from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import sys
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, cast

from tqdm import tqdm

from .io import resolve_experiment_output_path, validate_output_template, write_output

_LOGGER_NAME = "xgrid.runner"
_LOG_FORMAT = "%(levelname)s %(message)s"
_LOG_HANDLER_NAME = "xgrid.runner.stderr"

_MAX_PROGRESS_TEXT_LENGTH = 120


def configure_logging(log_level: str) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not any(handler.get_name() == _LOG_HANDLER_NAME for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.set_name(_LOG_HANDLER_NAME)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(_parse_log_level(log_level))
    logger.propagate = False
    return logger


def _parse_log_level(log_level: str) -> int:
    level = getattr(logging, log_level.upper(), None)
    if isinstance(level, int):
        return level
    return logging.INFO


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in config: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Config root must be a JSON object: {path}")
    return cast(dict[str, Any], payload)


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
        raise SystemExit("Config must define at least one experiment under 'experiments'")

    typed_experiments: dict[str, dict[str, Any]] = {}
    for experiment_key, entry in experiments.items():
        if not isinstance(entry, dict):
            raise SystemExit(f"Config for experiment '{experiment_key}' must be an object")
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
        typed_experiments[experiment_key] = {"fn": fn_name.strip(), "bindings": dict(bindings)}
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
            f"Missing bindings for experiment '{experiment_key}': {', '.join(missing_required)}"
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


@dataclass(frozen=True)
class BoundVariableSpec:
    argument_name: str
    variable_key: str
    generator_name: str
    generator: Callable[..., Iterable[tuple[object, Mapping[str, object]]]]


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


def load_script_module(script_path: Path) -> ModuleType:
    resolved_script = script_path.expanduser().resolve()
    if not resolved_script.exists():
        raise SystemExit(f"Script not found: {resolved_script}")
    if not resolved_script.is_file():
        raise SystemExit(f"Script path is not a file: {resolved_script}")

    module_name = f"_xgrid_script_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, resolved_script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load script: {resolved_script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    added_path = str(resolved_script.parent)
    sys.path.insert(0, added_path)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise SystemExit(f"Failed to import script: {resolved_script}: {exc}") from exc
    finally:
        if sys.path and sys.path[0] == added_path:
            del sys.path[0]
        else:
            try:
                sys.path.remove(added_path)
            except ValueError:
                pass
    return module


def build_rows_with_stats_from_bound_variables(
    fn: Callable[..., Any],
    *,
    bound_variable_specs: list[BoundVariableSpec],
    config_vars: dict[str, dict[str, Any]],
    show_progress: bool | None,
    logger: logging.Logger | None,
) -> tuple[list[dict[str, Any]], int]:
    show_progress_resolved = _resolve_show_progress(show_progress)
    progress_total: int | None = None
    if show_progress_resolved:
        progress_total = _compute_total_iterations(
            bound_variable_specs=bound_variable_specs,
            config_vars=config_vars,
        )
    if logger is not None:
        variable_summary = (
            ", ".join(
                f"{spec.argument_name}->{spec.variable_key}" for spec in bound_variable_specs
            )
            or "none"
        )
        if progress_total is None:
            logger.info(
                "Initialized lazy variable iteration variables=%s total_iterations=unknown",
                variable_summary,
            )
        else:
            logger.info(
                "Initialized lazy variable iteration variables=%s total_iterations=%d",
                variable_summary,
                progress_total,
            )

    rows: list[dict[str, Any]] = []
    iteration_count = 0
    with tqdm(
        total=progress_total,
        disable=not show_progress_resolved,
        dynamic_ncols=True,
    ) as progress:
        for values, meta in _iter_variable_combinations(
            bound_variable_specs=bound_variable_specs,
            config_vars=config_vars,
        ):
            if show_progress_resolved:
                progress.set_postfix_str(_format_progress_metadata(meta))
            result = fn(**values)
            for row in _normalize_result(result):
                combined = _merge_row(meta, row)
                rows.append(combined)
            iteration_count += 1
            progress.update(1)
    return rows, iteration_count


def iter_variable(
    spec: BoundVariableSpec, config: dict[str, Any]
) -> Iterable[tuple[Any, dict[str, Any]]]:
    kwargs = _filter_kwargs(spec.generator, config)
    for item in spec.generator(**kwargs):
        if not isinstance(item, tuple) or len(item) != 2:
            raise SystemExit(
                f"Variable '{spec.variable_key}' bound to argument "
                f"'{spec.argument_name}' must yield (value, metadata)"
            )
        value, metadata = item
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise SystemExit(
                f"Metadata for variable '{spec.variable_key}' bound to argument "
                f"'{spec.argument_name}' must be a dict"
            )
        yield value, cast(dict[str, Any], metadata)


def _filter_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(fn)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return dict(kwargs)
    allowed = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in allowed}


def _normalize_result(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        return [result]
    if isinstance(result, list) and all(isinstance(item, dict) for item in result):
        return cast(list[dict[str, Any]], result)
    raise SystemExit("Experiment must return a dict or list of dicts")


def _merge_row(*sources: dict[str, Any]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for source in sources:
        for key, value in source.items():
            if key in combined:
                raise SystemExit(f"Duplicate key in row: '{key}'")
            combined[key] = value
    return combined


def _resolve_show_progress(show_progress: bool | None) -> bool:
    if show_progress is not None:
        return show_progress
    return sys.stderr.isatty()


def _format_progress_metadata(metadata: dict[str, Any]) -> str:
    if not metadata:
        return ""
    parts = [f"{key}={str(value)}" for key, value in sorted(metadata.items())]
    return _truncate_text(", ".join(parts), _MAX_PROGRESS_TEXT_LENGTH)


def _truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return f"{text[: max_length - 3]}..."


def _count_variable_items(spec: BoundVariableSpec, config: dict[str, Any]) -> int:
    return sum(1 for _value, _metadata in iter_variable(spec, config))


def _compute_total_iterations(
    *,
    bound_variable_specs: list[BoundVariableSpec],
    config_vars: dict[str, dict[str, Any]],
) -> int:
    total = 1
    for spec in bound_variable_specs:
        variable_config = config_vars.get(spec.variable_key, {})
        count = _count_variable_items(spec, variable_config)
        total *= count
        if total == 0:
            return 0
    return total


def _iter_variable_combinations(
    *,
    bound_variable_specs: list[BoundVariableSpec],
    config_vars: dict[str, dict[str, Any]],
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    # Yields (values, metadata) where metadata uses <arg>__<key> prefixes.

    def _walk(
        index: int,
        values: dict[str, Any],
        metadata: dict[str, Any],
    ) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
        if index >= len(bound_variable_specs):
            yield dict(values), dict(metadata)
            return

        spec = bound_variable_specs[index]
        variable_config = config_vars.get(spec.variable_key, {})
        for value, item_metadata in iter_variable(spec, variable_config):
            values[spec.argument_name] = value
            added_keys: list[str] = []
            for key, meta_value in item_metadata.items():
                metadata_key = f"{spec.argument_name}__{key}"
                if metadata_key in metadata:
                    raise SystemExit(f"Duplicate metadata key in grid: '{metadata_key}'")
                metadata[metadata_key] = meta_value
                added_keys.append(metadata_key)
            yield from _walk(index + 1, values, metadata)
            for metadata_key in added_keys:
                del metadata[metadata_key]
            del values[spec.argument_name]

    yield from _walk(0, {}, {})


@dataclass(frozen=True)
class ExperimentReport:
    experiment_key: str
    fn_name: str
    output_path: Path
    rows_written: int
    iterations: int


@dataclass(frozen=True)
class RunReport:
    script_path: Path
    config_path: Path
    output_template: Path
    output_format: str | None
    experiments: list[ExperimentReport]
    experiment_catalog: list[dict[str, str]]
    results: dict[str, list[dict[str, Any]]]


def run_script_detailed(
    script_path: str | Path,
    *,
    config_path: str | Path,
    output_template: str | Path,
    output_format: str | None = None,
    show_progress: bool | None = None,
    log_level: str = "INFO",
) -> RunReport:
    logger = configure_logging(log_level)
    script_path_obj = Path(script_path)
    config_path_obj = Path(config_path)
    output_template_obj = Path(output_template)

    config = load_config(config_path_obj)
    if "environment" in config:
        raise SystemExit("Config key 'environment' is no longer supported; remove it.")

    experiments = validate_experiment_entries(config)
    variables = validate_variable_entries(config)
    validate_output_template(
        output_template=output_template_obj,
        experiments=experiments,
        output_format=output_format,
    )

    module = load_script_module(script_path_obj)

    warn_unused_variables(variables=variables, experiments=experiments)

    experiment_catalog = [
        {"key": experiment_key, "fn": entry["fn"]}
        for experiment_key, entry in experiments.items()
    ]

    results: dict[str, list[dict[str, Any]]] = {}
    reports: list[ExperimentReport] = []
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
        write_output(rows, output_path=output_path, output_format=output_format)
        logger.info(
            "Completed run experiment_key=%s iterations=%d rows_written=%d output=%s",
            experiment_key,
            total_iterations,
            len(rows),
            output_path,
        )
        results[experiment_key] = rows
        reports.append(
            ExperimentReport(
                experiment_key=experiment_key,
                fn_name=fn_name,
                output_path=output_path,
                rows_written=len(rows),
                iterations=total_iterations,
            )
        )

    return RunReport(
        script_path=script_path_obj,
        config_path=config_path_obj,
        output_template=output_template_obj,
        output_format=output_format,
        experiments=reports,
        experiment_catalog=experiment_catalog,
        results=results,
    )


def run_script(
    script_path: str | Path,
    *,
    config_path: str | Path,
    output_template: str | Path,
    output_format: str | None = None,
    show_progress: bool | None = None,
    log_level: str = "INFO",
) -> dict[str, list[dict[str, Any]]]:
    return run_script_detailed(
        script_path,
        config_path=config_path,
        output_template=output_template,
        output_format=output_format,
        show_progress=show_progress,
        log_level=log_level,
    ).results


def run_registered_experiment(
    fn: Callable[..., Any],
    *,
    config_path: Path,
    output_path: Path,
    output_format: str | None,
    show_progress: bool | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    if "environment" in config:
        raise SystemExit("Config key 'environment' is no longer supported; remove it.")
    experiments = validate_experiment_entries(config)
    variables = validate_variable_entries(config)

    matches = [key for key, entry in experiments.items() if entry.get("fn") == fn.__name__]
    if not matches:
        raise SystemExit(f"Config does not define an experiment for function '{fn.__name__}'")
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
        raise SystemExit(f"Unable to resolve module for experiment function '{fn.__name__}'")

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
    write_output(rows, output_path=output_path, output_format=output_format)
    if logger is not None:
        logger.info(
            "Completed run experiment=%s iterations=%d rows_written=%d output=%s",
            experiment_key,
            total_iterations,
            len(rows),
            output_path,
        )
    return rows


def build_rows(
    fn: Callable[..., Any],
    *,
    config: dict[str, Any],
    experiment_key: str | None = None,
    module: object | None = None,
    show_progress: bool | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    if "environment" in config:
        raise SystemExit("Config key 'environment' is no longer supported; remove it.")

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
        matches = [key for key, entry in experiments.items() if entry.get("fn") == fn.__name__]
        if not matches:
            raise SystemExit(f"Config does not define an experiment for function '{fn.__name__}'")
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

