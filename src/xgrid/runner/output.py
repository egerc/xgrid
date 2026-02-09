from __future__ import annotations

from pathlib import Path
from typing import Mapping

_EXPERIMENT_PLACEHOLDER = "{experiment}"
_OUTPUT_EXTENSIONS: dict[str, str] = {
    "csv": ".csv",
    "jsonl": ".jsonl",
    "parquet": ".parquet",
}


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
        return output_template

    if output_format is None:
        raise SystemExit(
            "Directory outputs require --format (csv, jsonl, parquet), "
            "or include an output extension in --output."
        )
    ext = _OUTPUT_EXTENSIONS.get(output_format)
    if ext is None:
        raise SystemExit(f"Unsupported output format: {output_format}")
    return output_template / f"{experiment_key}{ext}"
