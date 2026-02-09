from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from types import ModuleType


def load_script_module(script_path: Path) -> ModuleType:
    resolved_script = script_path.expanduser().resolve()
    if not resolved_script.exists():
        raise SystemExit(f"Script not found: {resolved_script}")
    if not resolved_script.is_file():
        raise SystemExit(f"Script path is not a file: {resolved_script}")

    module_name = f"_xgrid_script_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, resolved_script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load script: {resolved_script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    added_path = str(resolved_script.parent)
    sys.path.insert(0, added_path)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise SystemExit(f"Failed to import script: {resolved_script}: {exc}") from exc
    finally:
        if sys.path and sys.path[0] == added_path:
            del sys.path[0]
        else:
            try:
                sys.path.remove(added_path)
            except ValueError:
                pass
    return module
