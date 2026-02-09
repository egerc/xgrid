from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..lock import materialize_lock
from ..models import EnvironmentSpec, PreparedEnvironment
from ..util import run_command


class UvBackend:
    name = "uv"

    def prepare(
        self,
        *,
        project_root: Path,
        spec: EnvironmentSpec,
        fingerprint: str,
        rebuild: bool,
        refresh_lock: bool,
        lock_override_content: str | None,
    ) -> PreparedEnvironment:
        uv_binary = shutil.which("uv")
        if uv_binary is None:
            raise SystemExit("Environment backend 'uv' requires the 'uv' command")

        cache_dir = project_root / ".xgrid" / "envs" / fingerprint
        if rebuild and cache_dir.exists():
            shutil.rmtree(cache_dir)

        lock_material = materialize_lock(
            project_root=project_root,
            cache_dir=cache_dir,
            spec=spec,
            refresh_lock=refresh_lock,
            lock_override_content=lock_override_content,
        )
        venv_dir = cache_dir / "venv"
        python_executable = venv_dir / "bin" / "python"
        needs_bootstrap = rebuild or not python_executable.exists()
        if needs_bootstrap:
            run_command(
                [uv_binary, "venv", "--python", spec.python, str(venv_dir)],
                cwd=project_root,
            )

        if needs_bootstrap or refresh_lock:
            run_command(
                [
                    uv_binary,
                    "pip",
                    "install",
                    "--python",
                    str(python_executable),
                    "-e",
                    str(project_root),
                ],
                cwd=project_root,
            )
            if lock_material.content.strip():
                run_command(
                    [
                        uv_binary,
                        "pip",
                        "install",
                        "--python",
                        str(python_executable),
                        "-r",
                        str(lock_material.path),
                    ],
                    cwd=project_root,
                )

        status = "rebuild" if rebuild else "bootstrap" if needs_bootstrap else "reuse"
        return PreparedEnvironment(
            backend="uv",
            status=status,
            python_version=spec.python,
            fingerprint=fingerprint,
            cache_dir=cache_dir,
            lock_material=lock_material,
            python_executable=python_executable,
        )

    def run(
        self,
        *,
        project_root: Path,
        prepared: PreparedEnvironment,
        run_arguments: list[str],
    ) -> None:
        if prepared.python_executable is None:
            raise SystemExit("Prepared uv environment is missing python executable")

        env = os.environ.copy()
        src_path = str(project_root / "src")
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            src_path if not existing else f"{src_path}{os.pathsep}{existing}"
        )
        command = [str(prepared.python_executable), "-m", "xgrid", *run_arguments]
        run_command(command, cwd=project_root, env=env)
