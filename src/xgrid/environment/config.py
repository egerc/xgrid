from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from .models import EnvironmentSpec, ParsedEnvironmentConfig

_CONFIG_BACKENDS = {"uv"}
_CLI_BACKENDS = {"auto", "none", "uv"}


def parse_environment_config(
    config: Mapping[str, Any], *, config_path: Path
) -> ParsedEnvironmentConfig:
    entry = config.get("environment")
    if entry is None:
        return ParsedEnvironmentConfig(
            backend=None,
            python=None,
            dependencies=(),
            requirements_files=(),
        )
    if not isinstance(entry, dict):
        raise SystemExit("Config 'environment' must be an object")

    if "docker" in entry:
        raise SystemExit(
            "Config 'environment.docker' is not supported (Docker backend was removed)"
        )

    backend = entry.get("backend")
    if backend is not None:
        if backend == "docker":
            raise SystemExit(
                "Docker backend was removed; use 'uv' or omit environment.backend."
            )
        if not isinstance(backend, str) or backend.strip() not in _CONFIG_BACKENDS:
            allowed = ", ".join(sorted(_CONFIG_BACKENDS))
            raise SystemExit(f"Config 'environment.backend' must be one of: {allowed}")
        backend = backend.strip()

    python_target = entry.get("python")
    if python_target is not None:
        if not isinstance(python_target, str) or not python_target.strip():
            raise SystemExit("Config 'environment.python' must be a non-empty string")
        python_target = python_target.strip()

    dependencies = _parse_string_list(
        entry.get("dependencies"),
        field_name="environment.dependencies",
    )
    requirements_files_raw = _parse_string_list(
        entry.get("requirements_files"),
        field_name="environment.requirements_files",
    )
    requirements_files = tuple(
        _resolve_requirements_path(config_path=config_path, value=value)
        for value in requirements_files_raw
    )

    return ParsedEnvironmentConfig(
        backend=backend,
        python=python_target,
        dependencies=dependencies,
        requirements_files=requirements_files,
    )


def select_backend(
    cli_backend: str, *, parsed_environment: ParsedEnvironmentConfig
) -> str:
    if cli_backend == "auto":
        if parsed_environment.backend is not None:
            return parsed_environment.backend
        return "uv"

    if cli_backend == "docker":
        raise SystemExit("Docker backend was removed; choose 'uv' or 'none'.")

    if cli_backend in _CLI_BACKENDS:
        return cli_backend
    allowed = ", ".join(sorted(_CLI_BACKENDS))
    raise SystemExit(f"Unknown environment backend '{cli_backend}'. Allowed: {allowed}")


def build_environment_spec(
    *,
    selected_backend: str,
    parsed_environment: ParsedEnvironmentConfig,
) -> EnvironmentSpec | None:
    if selected_backend == "none":
        return None
    if selected_backend == "docker":
        raise SystemExit("Docker backend was removed; choose 'uv' or 'none'.")
    if selected_backend != "uv":
        raise SystemExit(f"Unsupported environment backend: {selected_backend}")

    python_target = parsed_environment.python or _default_python_target()
    return EnvironmentSpec(
        backend="uv",
        python=python_target,
        dependencies=parsed_environment.dependencies,
        requirements_files=parsed_environment.requirements_files,
    )


def _parse_string_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SystemExit(f"Config '{field_name}' must be an array of strings")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SystemExit(
                f"Config '{field_name}' must contain only non-empty strings"
            )
        parsed.append(item.strip())
    return tuple(parsed)


def _resolve_requirements_path(*, config_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    if not path.exists():
        raise SystemExit(f"Requirements file not found: {path}")
    if not path.is_file():
        raise SystemExit(f"Requirements path is not a file: {path}")
    return path


def _default_python_target() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"
