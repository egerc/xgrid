from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

from .registry import VariableSpec, get_variable_registry


def run_experiment(fn: Callable[..., Any], *, variables: list[str] | None = None, argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = _parse_args(argv)
    config_path = _resolve_config_path(args.config)
    config = _load_config(config_path)
    rows = build_rows(fn, variables=variables, config=config)
    _write_output(rows, output_path=Path(args.output), output_format=args.format)
    return rows


def build_rows(fn: Callable[..., Any], *, variables: list[str] | None, config: dict[str, Any]) -> list[dict[str, Any]]:
    variable_specs = _resolve_variables(variables)
    config_vars = _validate_config(config, variable_specs)
    materialized = [_materialize_variable(spec, config_vars[spec.name]) for spec in variable_specs]

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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="xgrid experiment runner")
    parser.add_argument("--config", help="Path to config.json", default=None)
    parser.add_argument("--output", help="Output path (.csv, .jsonl, .parquet)", required=True)
    parser.add_argument("--format", choices=["csv", "jsonl", "parquet"], default=None)
    return parser.parse_args(argv)


def _resolve_config_path(config_arg: str | None) -> Path:
    if config_arg:
        return Path(config_arg)
    if sys.stdin.isatty():
        try:
            user_input = input("Config path [config.json]: ").strip()
        except EOFError:
            user_input = ""
        if not user_input:
            user_input = "config.json"
        return Path(user_input)
    raise SystemExit("Missing --config in non-interactive mode")


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in config: {path}") from exc


def _resolve_variables(variables: list[str] | None) -> list[VariableSpec[object, object]]:
    registry = get_variable_registry()
    if not registry:
        raise SystemExit("No variables registered. Use the @variable decorator.")
    if variables is None:
        return list(registry.values())
    missing = [name for name in variables if name not in registry]
    if missing:
        raise SystemExit(f"Unknown variables requested: {', '.join(missing)}")
    return [registry[name] for name in variables]


def _validate_config(config: dict[str, Any], variable_specs: list[VariableSpec[object, object]]) -> dict[str, dict[str, Any]]:
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


def _materialize_variable(spec: VariableSpec[object, object], config: dict[str, Any]) -> list[tuple[Any, dict[str, Any]]]:
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
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
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


def _write_output(rows: list[dict[str, Any]], *, output_path: Path, output_format: str | None) -> None:
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
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Parquet output requires pandas and pyarrow") from exc
    df = pd.DataFrame(rows)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise SystemExit("Parquet output requires pandas and pyarrow") from exc


def _collect_fieldnames(rows: Iterable[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames
