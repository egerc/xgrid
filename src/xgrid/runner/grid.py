from __future__ import annotations

import inspect
import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, cast

from tqdm import tqdm

_MAX_PROGRESS_TEXT_LENGTH = 120


@dataclass(frozen=True)
class BoundVariableSpec:
    argument_name: str
    variable_key: str
    generator_name: str
    generator: Callable[..., Iterable[tuple[object, Mapping[str, object]]]]


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
                    raise SystemExit(
                        f"Duplicate metadata key in grid: '{metadata_key}'"
                    )
                metadata[metadata_key] = meta_value
                added_keys.append(metadata_key)
            yield from _walk(index + 1, values, metadata)
            for metadata_key in added_keys:
                del metadata[metadata_key]
            del values[spec.argument_name]

    yield from _walk(0, {}, {})
