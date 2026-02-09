from __future__ import annotations

from pathlib import Path

from .backends.uv import UvBackend
from .config import build_environment_spec, parse_environment_config, select_backend
from .fingerprint import compute_environment_fingerprint
from .lock import materialize_lock
from .models import (
    EnvironmentSpec,
    LockMaterial,
    ParsedEnvironmentConfig,
    PreparedEnvironment,
    local_environment,
)

__all__ = [
    "EnvironmentSpec",
    "LockMaterial",
    "ParsedEnvironmentConfig",
    "PreparedEnvironment",
    "build_environment_spec",
    "compute_environment_fingerprint",
    "materialize_lock",
    "parse_environment_config",
    "prepare_environment",
    "run_in_prepared_environment",
    "select_backend",
]

_UV_BACKEND = UvBackend()
_MANAGED_BACKENDS = {"uv": _UV_BACKEND}


def prepare_environment(
    *,
    project_root: Path,
    spec: EnvironmentSpec | None,
    fingerprint: str | None,
    rebuild: bool,
    refresh_lock: bool,
    lock_override_content: str | None = None,
) -> PreparedEnvironment:
    if spec is None:
        return local_environment()

    if not fingerprint:
        raise SystemExit("Missing environment fingerprint for managed backend")

    backend = _MANAGED_BACKENDS.get(spec.backend)
    if backend is None:
        raise SystemExit(f"Unsupported environment backend: {spec.backend}")
    return backend.prepare(
        project_root=project_root,
        spec=spec,
        fingerprint=fingerprint,
        rebuild=rebuild,
        refresh_lock=refresh_lock,
        lock_override_content=lock_override_content,
    )


def run_in_prepared_environment(
    *,
    project_root: Path,
    prepared: PreparedEnvironment,
    run_arguments: list[str],
) -> None:
    if prepared.backend == "uv":
        _UV_BACKEND.run(
            project_root=project_root,
            prepared=prepared,
            run_arguments=run_arguments,
        )
        return

    raise SystemExit(f"Unsupported prepared backend: {prepared.backend}")
