from __future__ import annotations

import tomllib
from pathlib import Path

from .models import EnvironmentSpec, LockMaterial
from .util import sha256_text


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
            fingerprint=sha256_text(content),
        )

    if lock_path.exists() and not refresh_lock:
        content = lock_path.read_text()
        return LockMaterial(
            path=lock_path,
            content=content,
            fingerprint=sha256_text(content),
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
        fingerprint=sha256_text(content),
    )


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


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _normalize_lock_content(content: str) -> str:
    normalized = content.strip()
    if not normalized:
        return ""
    return f"{normalized}\n"
