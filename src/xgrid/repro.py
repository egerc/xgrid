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
    output_path: Path
    output_format: str | None
    experiment_name: str | None
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
    selected_backend: str,
    environment_fingerprint: str | None,
    lock_fingerprint: str | None,
    lock_material: str | None,
    python_version: str,
    environment_status: str,
    xgrid_version: str,
    normalized_cli_argv: list[str],
) -> Path:
    manifest_path = sidecar_path_for_output(context.output_path)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "xgrid_version": xgrid_version,
        "run": {
            "script_path": str(context.script_path),
            "config_path": str(context.config_path),
            "output_path": str(context.output_path),
            "output_format": context.output_format,
            "experiment_name": context.experiment_name,
            "show_progress": context.show_progress,
            "log_level": context.log_level,
        },
        "hashes": {
            "script_sha256": sha256_file(context.script_path),
            "config_sha256": sha256_file(context.config_path),
        },
        "environment": {
            "backend": selected_backend,
            "fingerprint": environment_fingerprint,
            "lock_fingerprint": lock_fingerprint,
            "python": python_version,
            "status": environment_status,
            "lock_material": lock_material,
        },
        "cli": {"argv": normalized_cli_argv},
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return manifest_path
