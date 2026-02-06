from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def resolve_manifest_path(target: Path) -> Path:
    resolved = target.expanduser().resolve()
    if resolved.name.endswith(".run.json"):
        return resolved
    return resolved.with_name(f"{resolved.name}.run.json")


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


def read_run_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Run manifest not found: {path}")
    if not path.is_file():
        raise SystemExit(f"Run manifest path is not a file: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in run manifest: {path}") from exc

    if not isinstance(payload, dict):
        raise SystemExit(f"Run manifest must be a JSON object: {path}")
    _validate_manifest(payload, path=path)
    return payload


def context_from_manifest(payload: dict[str, Any]) -> RunContext:
    run = payload["run"]
    return RunContext(
        script_path=Path(run["script_path"]).expanduser().resolve(),
        config_path=Path(run["config_path"]).expanduser().resolve(),
        output_path=Path(run["output_path"]).expanduser().resolve(),
        output_format=run.get("output_format"),
        experiment_name=run.get("experiment_name"),
        show_progress=run.get("show_progress"),
        log_level=run.get("log_level", "INFO"),
    )


def validate_manifest_inputs(payload: dict[str, Any]) -> None:
    run = payload["run"]
    hashes = payload["hashes"]
    script_path = Path(run["script_path"]).expanduser().resolve()
    config_path = Path(run["config_path"]).expanduser().resolve()
    script_hash = sha256_file(script_path)
    config_hash = sha256_file(config_path)
    if script_hash != hashes["script_sha256"]:
        raise SystemExit(
            "Script content changed since manifest creation. "
            "Pass --allow-drift to rerun anyway."
        )
    if config_hash != hashes["config_sha256"]:
        raise SystemExit(
            "Config content changed since manifest creation. "
            "Pass --allow-drift to rerun anyway."
        )


def _validate_manifest(payload: dict[str, Any], *, path: Path) -> None:
    required_top_level = {"run", "hashes", "environment"}
    missing = required_top_level - set(payload.keys())
    if missing:
        raise SystemExit(
            f"Run manifest missing keys ({', '.join(sorted(missing))}): {path}"
        )

    run = payload["run"]
    if not isinstance(run, dict):
        raise SystemExit(f"Run manifest key 'run' must be an object: {path}")
    for key in ("script_path", "config_path", "output_path"):
        if key not in run or not isinstance(run[key], str) or not run[key].strip():
            raise SystemExit(f"Run manifest missing run.{key}: {path}")

    hashes = payload["hashes"]
    if not isinstance(hashes, dict):
        raise SystemExit(f"Run manifest key 'hashes' must be an object: {path}")
    for key in ("script_sha256", "config_sha256"):
        if key not in hashes or not isinstance(hashes[key], str) or not hashes[key]:
            raise SystemExit(f"Run manifest missing hashes.{key}: {path}")

    environment = payload["environment"]
    if not isinstance(environment, dict):
        raise SystemExit(f"Run manifest key 'environment' must be an object: {path}")
    backend = environment.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        raise SystemExit(f"Run manifest missing environment.backend: {path}")
