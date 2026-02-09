from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

_ALLOWED_BACKENDS = {"none", "uv", "docker"}


@dataclass(frozen=True)
class ParsedEnvironmentConfig:
    backend: str | None
    python: str | None
    dependencies: tuple[str, ...]
    requirements_files: tuple[Path, ...]
    docker_base_image: str | None


@dataclass(frozen=True)
class EnvironmentSpec:
    backend: Literal["uv", "docker"]
    python: str
    dependencies: tuple[str, ...]
    requirements_files: tuple[Path, ...]
    docker_base_image: str


@dataclass(frozen=True)
class LockMaterial:
    path: Path
    content: str
    fingerprint: str


@dataclass(frozen=True)
class PreparedEnvironment:
    backend: str
    status: str
    python_version: str
    fingerprint: str | None = None
    cache_dir: Path | None = None
    lock_material: LockMaterial | None = None
    python_executable: Path | None = None
    image_tag: str | None = None


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
            docker_base_image=None,
        )
    if not isinstance(entry, dict):
        raise SystemExit("Config 'environment' must be an object")

    backend = entry.get("backend")
    if backend is not None:
        if not isinstance(backend, str) or backend not in _ALLOWED_BACKENDS - {"none"}:
            allowed = ", ".join(sorted(_ALLOWED_BACKENDS - {"none"}))
            raise SystemExit(f"Config 'environment.backend' must be one of: {allowed}")

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

    docker_entry = entry.get("docker")
    docker_base_image: str | None = None
    if docker_entry is not None:
        if not isinstance(docker_entry, dict):
            raise SystemExit("Config 'environment.docker' must be an object")
        docker_base = docker_entry.get("base_image")
        if docker_base is not None:
            if not isinstance(docker_base, str) or not docker_base.strip():
                raise SystemExit(
                    "Config 'environment.docker.base_image' must be a non-empty string"
                )
            docker_base_image = docker_base.strip()

    return ParsedEnvironmentConfig(
        backend=backend,
        python=python_target,
        dependencies=dependencies,
        requirements_files=requirements_files,
        docker_base_image=docker_base_image,
    )


def select_backend(
    cli_backend: str, *, parsed_environment: ParsedEnvironmentConfig
) -> str:
    if cli_backend == "auto":
        if parsed_environment.backend is not None:
            return parsed_environment.backend
        return "uv"
    if cli_backend in _ALLOWED_BACKENDS:
        return cli_backend
    allowed = ", ".join(sorted(_ALLOWED_BACKENDS | {"auto"}))
    raise SystemExit(f"Unknown environment backend '{cli_backend}'. Allowed: {allowed}")


def build_environment_spec(
    *,
    selected_backend: str,
    parsed_environment: ParsedEnvironmentConfig,
) -> EnvironmentSpec | None:
    if selected_backend == "none":
        return None
    if selected_backend not in {"uv", "docker"}:
        raise SystemExit(f"Unsupported environment backend: {selected_backend}")
    python_target = parsed_environment.python or _default_python_target()
    docker_base_image = (
        parsed_environment.docker_base_image or f"python:{python_target}-slim"
    )
    backend = "uv" if selected_backend == "uv" else "docker"
    return EnvironmentSpec(
        backend=backend,
        python=python_target,
        dependencies=parsed_environment.dependencies,
        requirements_files=parsed_environment.requirements_files,
        docker_base_image=docker_base_image,
    )


def compute_environment_fingerprint(
    *,
    spec: EnvironmentSpec,
) -> str:
    payload = {
        "backend": spec.backend,
        "python": spec.python,
        "dependencies": list(spec.dependencies),
        "docker_base_image": spec.docker_base_image,
        "requirements_files": [
            {"path": str(path), "sha256": _sha256_path(path)}
            for path in spec.requirements_files
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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
        return PreparedEnvironment(
            backend="none",
            status="local",
            python_version=sys.version.split()[0],
            fingerprint=None,
            cache_dir=None,
            lock_material=None,
            python_executable=Path(sys.executable),
            image_tag=None,
        )

    if not fingerprint:
        raise SystemExit("Missing environment fingerprint for managed backend")

    if spec.backend == "uv":
        return _prepare_uv_environment(
            project_root=project_root,
            spec=spec,
            fingerprint=fingerprint,
            rebuild=rebuild,
            refresh_lock=refresh_lock,
            lock_override_content=lock_override_content,
        )
    if spec.backend == "docker":
        return _prepare_docker_environment(
            project_root=project_root,
            spec=spec,
            fingerprint=fingerprint,
            rebuild=rebuild,
            refresh_lock=refresh_lock,
            lock_override_content=lock_override_content,
        )
    raise SystemExit(f"Unsupported environment backend: {spec.backend}")


def run_in_prepared_environment(
    *,
    project_root: Path,
    prepared: PreparedEnvironment,
    run_arguments: list[str],
) -> None:
    if prepared.backend == "uv":
        if prepared.python_executable is None:
            raise SystemExit("Prepared uv environment is missing python executable")
        env = os.environ.copy()
        src_path = str(project_root / "src")
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            src_path if not existing else f"{src_path}{os.pathsep}{existing}"
        )
        command = [str(prepared.python_executable), "-m", "xgrid", *run_arguments]
        _run_command(command, cwd=project_root, env=env)
        return

    if prepared.backend == "docker":
        if prepared.image_tag is None:
            raise SystemExit("Prepared docker environment is missing image tag")
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{project_root}:/workspace",
            "-w",
            "/workspace",
            "-e",
            "PYTHONPATH=/workspace/src",
            prepared.image_tag,
            "python",
            "-m",
            "xgrid",
            *run_arguments,
        ]
        _run_command(command, cwd=project_root)
        return

    raise SystemExit(f"Unsupported prepared backend: {prepared.backend}")


def rewrite_path_for_docker(path: Path, *, project_root: Path) -> Path:
    resolved_path = path.expanduser().resolve()
    resolved_root = project_root.expanduser().resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise SystemExit(
            "Docker backend requires script/config/output paths inside the current workspace"
        ) from exc
    return Path("/workspace") / relative


def _prepare_uv_environment(
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
        _run_command(
            [uv_binary, "venv", "--python", spec.python, str(venv_dir)],
            cwd=project_root,
        )

    if needs_bootstrap or refresh_lock:
        _run_command(
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
            _run_command(
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
        image_tag=None,
    )


def _prepare_docker_environment(
    *,
    project_root: Path,
    spec: EnvironmentSpec,
    fingerprint: str,
    rebuild: bool,
    refresh_lock: bool,
    lock_override_content: str | None,
) -> PreparedEnvironment:
    if shutil.which("docker") is None:
        raise SystemExit("Environment backend 'docker' requires the 'docker' command")

    cache_dir = project_root / ".xgrid" / "envs" / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_material = materialize_lock(
        project_root=project_root,
        cache_dir=cache_dir,
        spec=spec,
        refresh_lock=refresh_lock,
        lock_override_content=lock_override_content,
    )
    dockerfile_path = cache_dir / "Dockerfile"
    dockerfile_path.write_text(_build_dockerfile(spec=spec))
    image_tag = f"xgrid-env:{fingerprint}"

    image_exists = _docker_image_exists(image_tag)
    should_build = rebuild or not image_exists or refresh_lock
    if should_build:
        _run_command(
            [
                "docker",
                "build",
                "-f",
                str(dockerfile_path),
                "-t",
                image_tag,
                str(cache_dir),
            ],
            cwd=project_root,
        )

    status = "rebuild" if rebuild else "bootstrap" if should_build else "reuse"
    return PreparedEnvironment(
        backend="docker",
        status=status,
        python_version=spec.python,
        fingerprint=fingerprint,
        cache_dir=cache_dir,
        lock_material=lock_material,
        python_executable=None,
        image_tag=image_tag,
    )


def materialize_lock(
    *,
    project_root: Path,
    cache_dir: Path,
    spec: EnvironmentSpec,
    refresh_lock: bool,
    lock_override_content: str | None = None,
) -> LockMaterial:
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / "requirements.lock"

    if lock_override_content is not None:
        content = _normalize_lock_content(lock_override_content)
        lock_path.write_text(content)
        return LockMaterial(
            path=lock_path,
            content=content,
            fingerprint=_sha256_text(content),
        )

    if lock_path.exists() and not refresh_lock:
        content = lock_path.read_text()
        return LockMaterial(
            path=lock_path,
            content=content,
            fingerprint=_sha256_text(content),
        )

    lines: list[str] = []
    for dependency in _project_runtime_dependencies(project_root):
        _append_unique(lines, dependency)
    for dependency in spec.dependencies:
        _append_unique(lines, dependency)
    for requirements_file in spec.requirements_files:
        for requirement_line in _iter_requirements_lines(requirements_file):
            _append_unique(lines, requirement_line)

    content = _normalize_lock_content("\n".join(lines))
    lock_path.write_text(content)
    return LockMaterial(
        path=lock_path,
        content=content,
        fingerprint=_sha256_text(content),
    )


def _build_dockerfile(*, spec: EnvironmentSpec) -> str:
    return "\n".join(
        [
            f"FROM {spec.docker_base_image}",
            "WORKDIR /workspace",
            "COPY requirements.lock /tmp/requirements.lock",
            "RUN python -m pip install --upgrade pip",
            "RUN if [ -s /tmp/requirements.lock ]; then "
            "python -m pip install -r /tmp/requirements.lock; fi",
        ]
    )


def _docker_image_exists(image_tag: str) -> bool:
    completed = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


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


def _project_runtime_dependencies(project_root: Path) -> list[str]:
    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return []
    try:
        data = tomllib.loads(pyproject.read_text())
    except Exception:
        return []
    project = data.get("project")
    if not isinstance(project, dict):
        return []
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list):
        return []
    parsed: list[str] = []
    for dependency in dependencies:
        if isinstance(dependency, str) and dependency.strip():
            parsed.append(dependency.strip())
    return parsed


def _iter_requirements_lines(path: Path) -> list[str]:
    parsed: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parsed.append(stripped)
    return parsed


def _default_python_target() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if completed.returncode != 0:
        rendered = " ".join(command)
        raise SystemExit(f"Command failed ({completed.returncode}): {rendered}")


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_lock_content(content: str) -> str:
    normalized = content.strip()
    if not normalized:
        return ""
    return f"{normalized}\n"
