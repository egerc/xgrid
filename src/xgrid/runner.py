from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import logging
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

from .registry import (
    ExperimentSpec,
    VariableSpec,
    _clear_registry,
    get_experiment_registry,
    get_variable_registry,
)

_LOGGER_NAME = "xgrid.runner"
_LOG_FORMAT = "%(levelname)s %(message)s"
_MAX_PROGRESS_TEXT_LENGTH = 120
_LOG_HANDLER_NAME = "xgrid.runner.stderr"


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
    rows, total_iterations = _build_rows_with_stats(
        fn,
        config=config,
        show_progress=show_progress,
        logger=logger,
    )
    _write_output(rows, output_path=output_path, output_format=output_format)
    if logger is not None:
        logger.info(
            "Completed run iterations=%d rows_written=%d output=%s",
            total_iterations,
            len(rows),
            output_path,
        )
    return rows


def run_script(
    script_path: str | Path,
    *,
    config_path: str | Path,
    output_path: str | Path,
    output_format: str | None = None,
    experiment_name: str | None = None,
    show_progress: bool | None = None,
    log_level: str = "INFO",
) -> list[dict[str, Any]]:
    logger = configure_logging(log_level)
    script_path_obj = Path(script_path)
    config_path_obj = Path(config_path)
    output_path_obj = Path(output_path)
    _clear_registry()
    _load_script_module(script_path_obj)
    experiment = _resolve_experiment(experiment_name)
    logger.info(
        "Starting run script=%s config=%s output=%s experiment=%s",
        script_path_obj,
        config_path_obj,
        output_path_obj,
        experiment.name,
    )
    return run_registered_experiment(
        experiment.fn,
        config_path=config_path_obj,
        output_path=output_path_obj,
        output_format=output_format,
        show_progress=show_progress,
        logger=logger,
    )


def _load_script_module(script_path: Path) -> None:
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


def _resolve_experiment(experiment_name: str | None) -> ExperimentSpec:
    experiments = get_experiment_registry()
    if not experiments:
        raise SystemExit("No experiments registered. Use the @experiment decorator.")

    available_names = sorted(experiments.keys())
    if experiment_name is not None:
        experiment = experiments.get(experiment_name)
        if experiment is None:
            available = ", ".join(available_names)
            raise SystemExit(
                f"Unknown experiment '{experiment_name}'. Available experiments: {available}"
            )
        return experiment

    if len(experiments) == 1:
        return experiments[available_names[0]]

    available = ", ".join(available_names)
    raise SystemExit(
        f"Multiple experiments found. Provide --experiment. Available experiments: {available}"
    )


def build_rows(
    fn: Callable[..., Any],
    *,
    config: dict[str, Any],
    show_progress: bool | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    rows, _ = _build_rows_with_stats(
        fn,
        config=config,
        show_progress=show_progress,
        logger=logger,
    )
    return rows


def _build_rows_with_stats(
    fn: Callable[..., Any],
    *,
    config: dict[str, Any],
    show_progress: bool | None,
    logger: logging.Logger | None,
) -> tuple[list[dict[str, Any]], int]:
    variable_specs = _resolve_variables()
    config_vars = _validate_config(config, variable_specs)
    show_progress_resolved = _resolve_show_progress(show_progress)
    progress_total: int | None = None
    if show_progress_resolved:
        progress_total = _compute_total_iterations(
            variable_specs=variable_specs,
            config_vars=config_vars,
        )
    if logger is not None:
        variable_summary = ", ".join(spec.name for spec in variable_specs) or "none"
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
            variable_specs=variable_specs,
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


def _resolve_variables() -> list[VariableSpec[object, object]]:
    registry = get_variable_registry()
    if not registry:
        raise SystemExit("No variables registered. Use the @variable decorator.")
    return list(registry.values())


def _validate_config(
    config: dict[str, Any], variable_specs: list[VariableSpec[object, object]]
) -> dict[str, dict[str, Any]]:
    variables = config.get("variables")
    if not isinstance(variables, dict):
        raise SystemExit("Config must contain a 'variables' object")

    missing = []
    for spec in variable_specs:
        if spec.config_key not in variables:
            if spec.name == spec.config_key:
                missing.append(spec.name)
            else:
                missing.append(f"{spec.name} (config key: {spec.config_key})")
    if missing:
        raise SystemExit(f"Missing variable configs: {', '.join(missing)}")

    variable_config_keys = {spec.config_key for spec in variable_specs}
    extra = [name for name in variables.keys() if name not in variable_config_keys]
    if extra:
        warnings.warn(f"Unknown variables in config: {', '.join(extra)}", stacklevel=2)

    typed_vars: dict[str, dict[str, Any]] = {}
    for spec in variable_specs:
        entry = variables.get(spec.config_key)
        if not isinstance(entry, dict):
            if spec.name == spec.config_key:
                raise SystemExit(f"Config for variable '{spec.name}' must be an object")
            raise SystemExit(
                f"Config for variable '{spec.name}' (config key: '{spec.config_key}') "
                "must be an object"
            )
        typed_vars[spec.name] = entry
    return typed_vars


def _iter_variable(
    spec: VariableSpec[object, object], config: dict[str, Any]
) -> Iterable[tuple[Any, dict[str, Any]]]:
    kwargs = _filter_kwargs(spec.generator, config)
    for item in spec.generator(**kwargs):
        if not isinstance(item, tuple) or len(item) != 2:
            raise SystemExit(f"Variable '{spec.name}' must yield (value, metadata)")
        value, metadata = item
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise SystemExit(f"Metadata for variable '{spec.name}' must be a dict")
        yield value, metadata


def _count_variable_items(
    spec: VariableSpec[object, object], config: dict[str, Any]
) -> int:
    return sum(1 for _value, _metadata in _iter_variable(spec, config))


def _compute_total_iterations(
    *,
    variable_specs: list[VariableSpec[object, object]],
    config_vars: dict[str, dict[str, Any]],
) -> int:
    total_iterations = 1
    for spec in variable_specs:
        total_iterations *= _count_variable_items(spec, config_vars[spec.name])
    return total_iterations


def _iter_variable_combinations(
    *,
    variable_specs: list[VariableSpec[object, object]],
    config_vars: dict[str, dict[str, Any]],
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    def _walk(
        index: int, values: dict[str, Any], metadata: dict[str, Any]
    ) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
        if index == len(variable_specs):
            yield dict(values), dict(metadata)
            return

        spec = variable_specs[index]
        for value, item_metadata in _iter_variable(spec, config_vars[spec.name]):
            values[spec.name] = value
            added_keys: list[str] = []
            for key, val in item_metadata.items():
                metadata_key = f"{spec.name}__{key}"
                metadata[metadata_key] = val
                added_keys.append(metadata_key)
            yield from _walk(index + 1, values, metadata)
            for metadata_key in added_keys:
                del metadata[metadata_key]
            del values[spec.name]

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
