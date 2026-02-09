import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xgrid import main as xgrid_main
from xgrid import repro


class CliReproTests(unittest.TestCase):
    def _write_script(self, path: Path) -> None:
        path.write_text(
            textwrap.dedent(
                """
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                def run(a: int):
                    return {"value": a}
                """
            )
        )

    def _write_config(self, path: Path, *, stop: int = 2) -> None:
        path.write_text(
            json.dumps(
                {
                    "variables": {
                        "gen_a": {"generator": "gen_a", "start": 0, "stop": stop}
                    },
                    "experiments": {"run": {"fn": "run", "bindings": {"a": "gen_a"}}},
                }
            )
        )

    def test_run_writes_sidecar_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "result.jsonl"
            self._write_script(script_path)
            self._write_config(config_path)

            code = xgrid_main(
                [
                    "run",
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_path),
                ]
            )
            self.assertEqual(code, 0)

            sidecar_path = repro.sidecar_path_for_output(output_path)
            self.assertTrue(sidecar_path.exists())
            payload = json.loads(sidecar_path.read_text())
            self.assertEqual(payload["schema_version"], 3)
            self.assertNotIn("environment", payload)
            self.assertEqual(
                payload["run"]["output_template"], str(output_path.resolve())
            )
            self.assertEqual(payload["run"]["output_path"], str(output_path.resolve()))
            self.assertEqual(payload["run"]["experiment"]["key"], "run")
            self.assertEqual(payload["run"]["experiment"]["fn"], "run")
            self.assertEqual(
                payload["run"]["experiments"], [{"key": "run", "fn": "run"}]
            )

    def test_rerun_command_is_rejected(self) -> None:
        with self.assertRaises(SystemExit) as exc:
            xgrid_main(["rerun", "results.jsonl"])
        self.assertEqual(exc.exception.code, 2)

    def test_module_entrypoint_supports_python_dash_m(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "src")
        completed = subprocess.run(
            [sys.executable, "-m", "xgrid", "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage: xgrid", completed.stdout)

    def test_run_rejects_env_control_flags(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "result.jsonl"
            self._write_script(script_path)
            self._write_config(config_path)

            with self.assertRaises(SystemExit) as exc:
                xgrid_main(
                    [
                        "run",
                        str(script_path),
                        "--config",
                        str(config_path),
                        "--output",
                        str(output_path),
                        "--env-backend",
                        "none",
                    ]
                )
            self.assertEqual(exc.exception.code, 2)

    def test_invalid_environment_dependencies_type_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "result.jsonl"
            self._write_script(script_path)
            config_path.write_text(
                json.dumps(
                    {
                        "environment": {"dependencies": "numpy==2.2.0"},
                        "variables": {
                            "gen_a": {"generator": "gen_a", "start": 0, "stop": 1}
                        },
                        "experiments": {
                            "run": {"fn": "run", "bindings": {"a": "gen_a"}}
                        },
                    }
                )
            )

            with self.assertRaises(SystemExit) as exc:
                xgrid_main(
                    [
                        "run",
                        str(script_path),
                        "--config",
                        str(config_path),
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertEqual(
                str(exc.exception),
                "Config key 'environment' is no longer supported; remove it.",
            )


if __name__ == "__main__":
    unittest.main()
