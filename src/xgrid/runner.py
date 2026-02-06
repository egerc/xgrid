from __future__ import annotations

import csv
import importlib.util
import inspect
import itertools
import json
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

from .registry import (
    ExperimentSpec,
    VariableSpec,
    _clear_registry,
    get_experiment_registry,
    get_variable_registry,
)


def run_registered_experiment(
    fn: Callable[..., Any],
    *,
    variables: list[str] | None,
    config_path: Path,
    output_path: Path,
    output_format: str | None,
) -> list[dict[str, Any]]:
    config = _load_config(config_path)
    rows = build_rows(fn, variables=variables, config=config)
    _write_output(rows, output_path=output_path, output_format=output_format)
    return rows


def run_script(
    script_path: str | Path,
    *,
    config_path: str | Path,
    output_path: str | Path,
    output_format: str | None = None,
    experiment_name: str | None = None,
) -> list[dict[str, Any]]:
    config_path_obj = Path(config_path)
    output_path_obj = Path(output_path)
    _clear_registry()
    _load_script_module(Path(script_path))
    experiment = _resolve_experiment(experiment_name)
    return run_registered_experiment(
        experiment.fn,
        variables=experiment.variables,
        config_path=config_path_obj,
        output_path=output_path_obj,
        output_format=output_format,
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
        raise SystemExit(f"Failed to import script: {resolved_script}") from exc
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
    fn: Callable[..., Any], *, variables: list[str] | None, config: dict[str, Any]
) -> list[dict[str, Any]]:
    variable_specs = _resolve_variables(variables)
    config_vars = _validate_config(config, variable_specs)
    materialized = [
        _materialize_variable(spec, config_vars[spec.name]) for spec in variable_specs
    ]

    rows: list[dict[str, Any]] = []
    for combo in itertools.product(*materialized):
        values: dict[str, Any] = {}
        meta: dict[str, Any] = {}
        for spec, (value, metadata) in zip(variable_specs, combo):
            values[spec.name] = value
            for key, val in metadata.items():
                meta[f"{spec.name}__{key}"] = val
        result = fn(**values)
        for row in _normalize_result(result):
            combined = _merge_row(meta, row)
            rows.append(combined)
    return rows


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in config: {path}") from exc


def _resolve_variables(
    variables: list[str] | None,
) -> list[VariableSpec[object, object]]:
    registry = get_variable_registry()
    if not registry:
        raise SystemExit("No variables registered. Use the @variable decorator.")
    if variables is None:
        return list(registry.values())
    missing = [name for name in variables if name not in registry]
    if missing:
        raise SystemExit(f"Unknown variables requested: {', '.join(missing)}")
    return [registry[name] for name in variables]


def _validate_config(
    config: dict[str, Any], variable_specs: list[VariableSpec[object, object]]
) -> dict[str, dict[str, Any]]:
    variables = config.get("variables")
    if not isinstance(variables, dict):
        raise SystemExit("Config must contain a 'variables' object")

    variable_names = {spec.name for spec in variable_specs}
    missing = [name for name in variable_names if name not in variables]
    if missing:
        raise SystemExit(f"Missing variable configs: {', '.join(missing)}")

    extra = [name for name in variables.keys() if name not in variable_names]
    if extra:
        warnings.warn(f"Unknown variables in config: {', '.join(extra)}", stacklevel=2)

    typed_vars: dict[str, dict[str, Any]] = {}
    for name in variable_names:
        entry = variables.get(name)
        if not isinstance(entry, dict):
            raise SystemExit(f"Config for variable '{name}' must be an object")
        typed_vars[name] = entry
    return typed_vars


def _materialize_variable(
    spec: VariableSpec[object, object], config: dict[str, Any]
) -> list[tuple[Any, dict[str, Any]]]:
    kwargs = _filter_kwargs(spec.generator, config)
    results = list(spec.generator(**kwargs))
    normalized: list[tuple[Any, dict[str, Any]]] = []
    for item in results:
        if not isinstance(item, tuple) or len(item) != 2:
            raise SystemExit(f"Variable '{spec.name}' must yield (value, metadata)")
        value, metadata = item
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise SystemExit(f"Metadata for variable '{spec.name}' must be a dict")
        normalized.append((value, metadata))
    return normalized


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
