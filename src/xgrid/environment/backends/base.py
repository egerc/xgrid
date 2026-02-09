from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import EnvironmentSpec, PreparedEnvironment


class ManagedEnvironmentBackend(Protocol):
    name: str

    def prepare(
        self,
        *,
        project_root: Path,
        spec: EnvironmentSpec,
        fingerprint: str,
        rebuild: bool,
        refresh_lock: bool,
        lock_override_content: str | None,
    ) -> PreparedEnvironment: ...

    def run(
        self,
        *,
        project_root: Path,
        prepared: PreparedEnvironment,
        run_arguments: list[str],
    ) -> None: ...
