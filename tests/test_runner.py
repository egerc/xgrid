import json
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xgrid import experiment, main as xgrid_main
from xgrid import registry as registry_module


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        registry_module._clear_registry()

    def _write_config(self, path: Path, variables: dict) -> None:
        path.write_text(json.dumps({"variables": variables}))

    def _write_script(self, path: Path, source: str) -> None:
        path.write_text(textwrap.dedent(source))

    def test_grid_generation_and_metadata_prefix_positional_script(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"

            self._write_script(
                script_path,
                """
                from xgrid import experiment, variable

                @variable(name="a")
                def gen_a(start: int, stop: int):
                    return [(i, {"value": i}) for i in range(start, stop)]

                @variable(name="b")
                def gen_b(start: int, stop: int):
                    return [(i, {"value": i}) for i in range(start, stop)]

                @experiment()
                def run(a: int, b: int):
                    return [{"sum": a + b}]
                """,
            )
            self._write_config(
                config_path,
                {
                    "a": {"start": 0, "stop": 2},
                    "b": {"start": 0, "stop": 3},
                },
            )

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

            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertNotIn("a", rows[0])
            self.assertNotIn("b", rows[0])
            self.assertIn("a__value", rows[0])
            self.assertIn("b__value", rows[0])

    def test_script_flag_form_and_dict_result(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"

            self._write_script(
                script_path,
                """
                from xgrid import experiment, variable

                @variable(name="a")
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment()
                def run(a: int):
                    return {"value": a}
                """,
            )
            self._write_config(config_path, {"a": {"start": 0, "stop": 2, "step": 1}})

            code = xgrid_main(
                [
                    "run",
                    "--script",
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_path),
                ]
            )
            self.assertEqual(code, 0)

            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(rows[0]["value"], 0)
            self.assertEqual(rows[1]["value"], 1)

    def test_csv_writer_inference(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.csv"

            self._write_script(
                script_path,
                """
                from xgrid import experiment, variable

                @variable(name="a")
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment()
                def run(a: int):
                    return {"value": a}
                """,
            )
            self._write_config(config_path, {"a": {"start": 0, "stop": 2}})

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

            header = output_path.read_text().splitlines()[0]
            header_fields = header.split(",")
            self.assertNotIn("a", header_fields)
            self.assertIn("value", header_fields)

    def test_multiple_experiments_requires_experiment(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"

            self._write_script(
                script_path,
                """
                from xgrid import experiment, variable

                @variable(name="a")
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment(variables=["a"])
                def first(a: int):
                    return {"kind": "first", "value": a}

                @experiment(variables=["a"])
                def second(a: int):
                    return {"kind": "second", "value": a}
                """,
            )
            self._write_config(config_path, {"a": {"start": 0, "stop": 1}})

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
            message = str(exc.exception)
            self.assertIn("Multiple experiments found", message)
            self.assertIn("first", message)
            self.assertIn("second", message)

    def test_multiple_experiments_valid_selection(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"

            self._write_script(
                script_path,
                """
                from xgrid import experiment, variable

                @variable(name="a")
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment(variables=["a"])
                def first(a: int):
                    return {"kind": "first", "value": a}

                @experiment(variables=["a"])
                def second(a: int):
                    return {"kind": "second", "value": a}
                """,
            )
            self._write_config(config_path, {"a": {"start": 0, "stop": 2}})

            xgrid_main(
                [
                    "run",
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_path),
                    "--experiment",
                    "second",
                ]
            )

            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["kind"] == "second" for row in rows))

    def test_unknown_experiment_lists_available(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"

            self._write_script(
                script_path,
                """
                from xgrid import experiment, variable

                @variable(name="a")
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment(variables=["a"])
                def first(a: int):
                    return {"kind": "first", "value": a}
                """,
            )
            self._write_config(config_path, {"a": {"start": 0, "stop": 1}})

            with self.assertRaises(SystemExit) as exc:
                xgrid_main(
                    [
                        "run",
                        str(script_path),
                        "--config",
                        str(config_path),
                        "--output",
                        str(output_path),
                        "--experiment",
                        "missing",
                    ]
                )
            message = str(exc.exception)
            self.assertIn("Unknown experiment", message)
            self.assertIn("first", message)

    def test_rejects_both_script_forms(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"

            self._write_script(
                script_path,
                """
                from xgrid import experiment, variable

                @variable(name="a")
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment(variables=["a"])
                def first(a: int):
                    return {"kind": "first", "value": a}
                """,
            )
            self._write_config(config_path, {"a": {"start": 0, "stop": 1}})

            with self.assertRaises(SystemExit) as exc:
                xgrid_main(
                    [
                        "run",
                        str(script_path),
                        "--script",
                        str(script_path),
                        "--config",
                        str(config_path),
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertIn("either positionally or with --script", str(exc.exception))

    def test_requires_config_argument(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            output_path = tmp_path / "out.jsonl"

            self._write_script(
                script_path,
                """
                from xgrid import experiment, variable

                @variable(name="a")
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment(variables=["a"])
                def first(a: int):
                    return {"kind": "first", "value": a}
                """,
            )

            with self.assertRaises(SystemExit) as exc:
                xgrid_main(["run", str(script_path), "--output", str(output_path)])
            self.assertEqual(exc.exception.code, 2)

    def test_direct_experiment_call_no_cli_side_effects(self) -> None:
        @experiment()
        def run_direct() -> dict:
            return {"ok": True}

        result = run_direct()
        self.assertEqual(result, {"ok": True})

    def test_parquet_requires_optional_deps(self) -> None:
        try:
            import polars  # noqa: F401
        except ImportError:
            polars = None

        if polars is not None:
            self.skipTest("polars installed; parquet behavior depends on optional deps")

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.parquet"

            self._write_script(
                script_path,
                """
                from xgrid import experiment, variable

                @variable(name="a")
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment(variables=["a"])
                def first(a: int):
                    return {"kind": "first", "value": a}
                """,
            )
            self._write_config(config_path, {"a": {"start": 0, "stop": 1}})

            with self.assertRaises(SystemExit):
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


if __name__ == "__main__":
    unittest.main()
