import json
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
                from xgrid import experiment, variable

                @variable
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment
                def run(a: int):
                    return {"value": a}
                """
            )
        )

    def _write_config(self, path: Path, *, stop: int = 2) -> None:
        path.write_text(
            json.dumps(
                {
                    "variables": {"gen_a": {"start": 0, "stop": stop}},
                    "experiments": {"run": {"bindings": {"a": "gen_a"}}},
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
                    "--env-backend",
                    "none",
                ]
            )
            self.assertEqual(code, 0)

            sidecar_path = repro.sidecar_path_for_output(output_path)
            self.assertTrue(sidecar_path.exists())
            payload = json.loads(sidecar_path.read_text())
            self.assertEqual(payload["environment"]["backend"], "none")
            self.assertEqual(payload["run"]["output_path"], str(output_path.resolve()))

    def test_rerun_from_output_path(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "result.jsonl"
            self._write_script(script_path)
            self._write_config(config_path, stop=3)

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
            output_path.unlink()
            code = xgrid_main(["rerun", str(output_path)])
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(len(rows), 3)

    def test_rerun_detects_drift_without_allow_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "result.jsonl"
            self._write_script(script_path)
            self._write_config(config_path, stop=2)
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

            self._write_config(config_path, stop=1)
            with self.assertRaises(SystemExit) as exc:
                xgrid_main(["rerun", str(output_path)])
            self.assertIn("Pass --allow-drift", str(exc.exception))

            code = xgrid_main(["rerun", str(output_path), "--allow-drift"])
            self.assertEqual(code, 0)

    def test_run_accepts_env_control_flags(self) -> None:
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
                    "--env-backend",
                    "none",
                    "--rebuild-env",
                    "--refresh-lock",
                ]
            )
            self.assertEqual(code, 0)

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
                        "variables": {"gen_a": {"start": 0, "stop": 1}},
                        "experiments": {"run": {"bindings": {"a": "gen_a"}}},
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
                        "--env-backend",
                        "none",
                    ]
                )
            self.assertIn(
                "environment.dependencies",
                str(exc.exception),
            )


if __name__ == "__main__":
    unittest.main()
