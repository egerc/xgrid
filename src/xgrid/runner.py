from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import logging
import sys
from types import ModuleType
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, cast

from tqdm import tqdm

_LOGGER_NAME = "xgrid.runner"
_LOG_FORMAT = "%(levelname)s %(message)s"
_MAX_PROGRESS_TEXT_LENGTH = 120
_LOG_HANDLER_NAME = "xgrid.runner.stderr"
_EXPERIMENT_PLACEHOLDER = "{experiment}"
_OUTPUT_EXTENSIONS: dict[str, str] = {
    "csv": ".csv",
    "jsonl": ".jsonl",
    "parquet": ".parquet",
}


@dataclass(frozen=True)
class BoundVariableSpec:
    argument_name: str
    variable_key: str
    generator_name: str
    generator: Callable[..., Iterable[tuple[object, Mapping[str, object]]]]


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


def validate_output_template(
    *,
    output_template: Path,
    experiments: Mapping[str, object],
    output_format: str | None,
) -> None:
    """
    Validate that --output can be resolved for the configured experiments.

    Supported modes:
    - File template mode: --output contains "{experiment}" and will be expanded per
      experiment.
    - Directory mode: --output is a directory (existing dir, or a path without a
      recognized file extension). Each experiment writes to <dir>/<experiment>.<ext>.
    - Single-file mode: --output is a single file path (recognized extension). Only
      allowed when there is exactly one experiment.
    """

    # Existing directory always means directory mode.
    if output_template.exists() and output_template.is_dir():
        if output_format is None:
            raise SystemExit(
                "Directory outputs require --format (csv, jsonl, parquet)."
            )
        return

    template_str = str(output_template)
    if _EXPERIMENT_PLACEHOLDER in template_str:
        return

    suffix = output_template.suffix.lower()
    if suffix in {".csv", ".jsonl", ".parquet"}:
        if len(experiments) > 1:
            raise SystemExit(
                "Multiple experiments in config require --output to be a directory "
                "or include '{experiment}'."
            )
        return

    # No placeholder and no known extension: treat as directory.
    if output_format is None:
        raise SystemExit(
            "Directory outputs require --format (csv, jsonl, parquet), "
            "or include an output extension in --output."
        )


def resolve_experiment_output_path(
    *,
    output_template: Path,
    experiment_key: str,
    output_format: str | None,
) -> Path:
    """
    Resolve the concrete output path for a single experiment.

    Directory mode writes to <output_template>/<experiment_key>.<ext>.
    Template mode replaces "{experiment}" in the output template.
    """

    if output_template.exists() and output_template.is_dir():
        if output_format is None:
            raise SystemExit(
                "Directory outputs require --format (csv, jsonl, parquet)."
            )
        ext = _OUTPUT_EXTENSIONS.get(output_format)
        if ext is None:
            raise SystemExit(f"Unsupported output format: {output_format}")
        return output_template / f"{experiment_key}{ext}"

    template_str = str(output_template)
    if _EXPERIMENT_PLACEHOLDER in template_str:
        return Path(template_str.replace(_EXPERIMENT_PLACEHOLDER, experiment_key))

    suffix = output_template.suffix.lower()
    if suffix in {".csv", ".jsonl", ".parquet"}:
        # Single-file mode.
        return output_template

    # Directory mode (path may not exist yet).
    if output_format is None:
        raise SystemExit(
            "Directory outputs require --format (csv, jsonl, parquet), "
            "or include an output extension in --output."
        )
    ext = _OUTPUT_EXTENSIONS.get(output_format)
    if ext is None:
        raise SystemExit(f"Unsupported output format: {output_format}")
    return output_template / f"{experiment_key}{ext}"


def run_registered_experiment(
    fn: Callable[..., Any],
    *,
    config_path: Path,
    output_path: Path,
    output_format: str | None,
    show_progress: bool | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    config = _load_config(config_path)
    experiments = _validate_experiment_entries(config)
    variables = _validate_variable_entries(config)

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

    _warn_unused_variables(variables=variables, experiments=experiments)
    bound_variable_specs, config_vars = _resolve_bound_variables(
        fn,
        experiment_key=experiment_key,
        bindings=experiment_entry["bindings"],
        variables=variables,
        module=module,
    )
    rows, total_iterations = _build_rows_with_stats_from_bound_variables(
        fn,
        bound_variable_specs=bound_variable_specs,
        config_vars=config_vars,
        show_progress=show_progress,
        logger=logger,
    )
    _write_output(rows, output_path=output_path, output_format=output_format)
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

    config = _load_config(config_path_obj)
    experiments = _validate_experiment_entries(config)
    variables = _validate_variable_entries(config)
    validate_output_template(
        output_template=output_template_obj,
        experiments=experiments,
        output_format=output_format,
    )

    module = _load_script_module(script_path_obj)

    _warn_unused_variables(variables=variables, experiments=experiments)

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
        bound_variable_specs, config_vars = _resolve_bound_variables(
            fn,
            experiment_key=experiment_key,
            bindings=experiment_entry["bindings"],
            variables=variables,
            module=module,
        )
        rows, total_iterations = _build_rows_with_stats_from_bound_variables(
            fn,
            bound_variable_specs=bound_variable_specs,
            config_vars=config_vars,
            show_progress=show_progress,
            logger=logger,
        )
        _write_output(rows, output_path=output_path, output_format=output_format)
        logger.info(
            "Completed run experiment_key=%s iterations=%d rows_written=%d output=%s",
            experiment_key,
            total_iterations,
            len(rows),
            output_path,
        )
        results[experiment_key] = rows
    return results


def _load_script_module(script_path: Path) -> ModuleType:
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


def build_rows(
    fn: Callable[..., Any],
    *,
    config: dict[str, Any],
    experiment_key: str | None = None,
    module: object | None = None,
    show_progress: bool | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    experiments = _validate_experiment_entries(config)
    variables = _validate_variable_entries(config)
    _warn_unused_variables(variables=variables, experiments=experiments)

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

    bound_variable_specs, config_vars = _resolve_bound_variables(
        fn,
        experiment_key=selected_key,
        bindings=entry["bindings"],
        variables=variables,
        module=resolved_module,
    )
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
                f"{spec.argument_name}->{spec.variable_key}"
                for spec in bound_variable_specs
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
    return rows


def _parse_log_level(log_level: str) -> int:
    level = getattr(logging, log_level.upper(), None)
    if isinstance(level, int):
        return level
    return logging.INFO


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


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in config: {path}") from exc


def _validate_variable_entries(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
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


def _validate_experiment_entries(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
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


def _resolve_bound_variables(
    fn: Callable[..., Any],
    *,
    experiment_key: str,
    bindings: dict[str, Any],
    variables: dict[str, dict[str, Any]],
    module: object,
) -> tuple[list[BoundVariableSpec], dict[str, dict[str, Any]]]:
    typed_bindings = _validate_experiment_bindings(
        fn,
        experiment_key=experiment_key,
        bindings=bindings,
    )

    bound_variable_specs: list[BoundVariableSpec] = []
    config_vars: dict[str, dict[str, Any]] = {}
    for argument_name, variable_key in typed_bindings.items():
        variable_entry = variables.get(variable_key)
        if variable_entry is None:
            raise SystemExit(
                f"Unknown variable '{variable_key}' bound to argument '{argument_name}' "
                f"for experiment '{experiment_key}'"
            )
        generator_name = str(variable_entry.get("generator"))
        generator_obj = getattr(module, generator_name, None)
        if not callable(generator_obj):
            raise SystemExit(
                f"Unknown generator function '{generator_name}' for variable '{variable_key}' "
                f"(experiment '{experiment_key}')"
            )
        generator = cast(
            Callable[..., Iterable[tuple[object, Mapping[str, object]]]],
            generator_obj,
        )
        variable_kwargs = {
            key: value for key, value in variable_entry.items() if key != "generator"
        }
        bound_variable_specs.append(
            BoundVariableSpec(
                argument_name=argument_name,
                variable_key=variable_key,
                generator_name=generator_name,
                generator=generator,
            )
        )
        config_vars[argument_name] = variable_kwargs

    return bound_variable_specs, config_vars


def _validate_experiment_bindings(
    fn: Callable[..., Any], *, experiment_key: str, bindings: dict[str, Any]
) -> dict[str, str]:
    signature = inspect.signature(fn)
    bindable_parameters: list[str] = []
    required_parameters: list[str] = []
    for parameter_name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            bindable_parameters.append(parameter_name)
            if parameter.default is inspect.Parameter.empty:
                required_parameters.append(parameter_name)

    bindable_set = set(bindable_parameters)
    unknown_bindings = [key for key in bindings.keys() if key not in bindable_set]
    if unknown_bindings:
        raise SystemExit(
            f"Unknown bindings for experiment '{experiment_key}': "
            f"{', '.join(sorted(unknown_bindings))}"
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


def _iter_variable(
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
        yield value, metadata


def _enforce_output_template(
    *,
    output_template: Path,
    experiments: dict[str, dict[str, Any]],
    output_format: str | None,
) -> None:
    # Backwards-compatible alias retained for internal callers.
    validate_output_template(
        output_template=output_template,
        experiments=experiments,
        output_format=output_format,
    )


def _resolve_output_path(
    *, output_template: Path, experiment_key: str, output_format: str | None
) -> Path:
    # Backwards-compatible alias retained for internal callers.
    return resolve_experiment_output_path(
        output_template=output_template,
        experiment_key=experiment_key,
        output_format=output_format,
    )


def _warn_unused_variables(
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


def _build_rows_with_stats_from_bound_variables(
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
                f"{spec.argument_name}->{spec.variable_key}"
                for spec in bound_variable_specs
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


def _count_variable_items(spec: BoundVariableSpec, config: dict[str, Any]) -> int:
    return sum(1 for _value, _metadata in _iter_variable(spec, config))


def _compute_total_iterations(
    *,
    bound_variable_specs: list[BoundVariableSpec],
    config_vars: dict[str, dict[str, Any]],
) -> int:
    total_iterations = 1
    for spec in bound_variable_specs:
        total_iterations *= _count_variable_items(spec, config_vars[spec.argument_name])
    return total_iterations


def _iter_variable_combinations(
    *,
    bound_variable_specs: list[BoundVariableSpec],
    config_vars: dict[str, dict[str, Any]],
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    def _walk(
        index: int, values: dict[str, Any], metadata: dict[str, Any]
    ) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
        if index == len(bound_variable_specs):
            yield dict(values), dict(metadata)
            return

        spec = bound_variable_specs[index]
        for value, item_metadata in _iter_variable(
            spec, config_vars[spec.argument_name]
        ):
            values[spec.argument_name] = value
            added_keys: list[str] = []
            for key, val in item_metadata.items():
                metadata_key = f"{spec.argument_name}__{key}"
                metadata[metadata_key] = val
                added_keys.append(metadata_key)
            yield from _walk(index + 1, values, metadata)
            for metadata_key in added_keys:
                del metadata[metadata_key]
            del values[spec.argument_name]

    yield from _walk(0, {}, {})


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
        return result
    raise SystemExit("Experiment must return a dict or list of dicts")


def _merge_row(*sources: dict[str, Any]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for source in sources:
        for key, value in source.items():
            if key in combined:
                raise SystemExit(f"Duplicate key in row: '{key}'")
            combined[key] = value
    return combined


def _write_output(
    rows: list[dict[str, Any]], *, output_path: Path, output_format: str | None
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_format or _infer_format(output_path)
    if fmt == "csv":
        _write_csv(output_path, rows)
        return
    if fmt == "jsonl":
        _write_jsonl(output_path, rows)
        return
    if fmt == "parquet":
        _write_parquet(output_path, rows)
        return
    raise SystemExit(f"Unsupported output format: {fmt}")


def _infer_format(output_path: Path) -> str:
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".parquet":
        return "parquet"
    raise SystemExit("Unable to infer format from output extension")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = _collect_fieldnames(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import polars as pl
    except ImportError as exc:
        raise SystemExit("Parquet output requires polars") from exc
    df = pl.DataFrame(rows)
    try:
        df.write_parquet(path)
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise SystemExit("Parquet output requires polars") from exc


def _collect_fieldnames(rows: Iterable[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames
