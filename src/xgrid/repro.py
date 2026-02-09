from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    script_path: Path
    config_path: Path
    output_template: Path
    output_format: str | None
    show_progress: bool | None
    log_level: str


def sidecar_path_for_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.run.json")


def sha256_file(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if not path.is_file():
        raise SystemExit(f"Path is not a file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_run_manifest(
    *,
    context: RunContext,
    output_path: Path,
    output_template: Path,
    experiment_key: str,
    experiment_fn: str,
    experiments: list[dict[str, str]],
    xgrid_version: str,
    normalized_cli_argv: list[str],
) -> Path:
    manifest_path = sidecar_path_for_output(output_path)
    payload = {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "xgrid_version": xgrid_version,
        "run": {
            "script_path": str(context.script_path),
            "config_path": str(context.config_path),
            "output_template": str(output_template),
            "output_path": str(output_path),
            "output_format": context.output_format,
            "show_progress": context.show_progress,
            "log_level": context.log_level,
            "experiment": {"key": experiment_key, "fn": experiment_fn},
            "experiments": experiments,
        },
        "hashes": {
            "script_sha256": sha256_file(context.script_path),
            "config_sha256": sha256_file(context.config_path),
        },
        "cli": {"argv": normalized_cli_argv},
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return manifest_path
