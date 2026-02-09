from __future__ import annotations

import hashlib
import json

from .models import EnvironmentSpec
from .util import sha256_path


def compute_environment_fingerprint(*, spec: EnvironmentSpec) -> str:
    payload = {
        "backend": spec.backend,
        "python": spec.python,
        "dependencies": list(spec.dependencies),
        "requirements_files": [
            {"path": str(path), "sha256": sha256_path(path)}
            for path in spec.requirements_files
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
