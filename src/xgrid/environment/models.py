from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ParsedEnvironmentConfig:
    backend: str | None
    python: str | None
    dependencies: tuple[str, ...]
    requirements_files: tuple[Path, ...]


@dataclass(frozen=True)
class EnvironmentSpec:
    backend: Literal["uv"]
    python: str
    dependencies: tuple[str, ...]
    requirements_files: tuple[Path, ...]


@dataclass(frozen=True)
class LockMaterial:
    path: Path
    content: str
    fingerprint: str


@dataclass(frozen=True)
class PreparedEnvironment:
    backend: Literal["none", "uv"]
    status: str
    python_version: str
    fingerprint: str | None = None
    cache_dir: Path | None = None
    lock_material: LockMaterial | None = None
    python_executable: Path | None = None


def local_environment() -> PreparedEnvironment:
    return PreparedEnvironment(
        backend="none",
        status="local",
        python_version=sys.version.split()[0],
        fingerprint=None,
        cache_dir=None,
        lock_material=None,
        python_executable=Path(sys.executable),
    )
