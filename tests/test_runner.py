import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xgrid import experiment, variable
from xgrid import registry as registry_module


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        registry_module._clear_registry()

    def _write_config(self, path: Path, variables: dict) -> None:
        path.write_text(json.dumps({"variables": variables}))

    def test_grid_generation_and_metadata_prefix(self) -> None:
        @variable(name="a")
        def gen_a(start: int, stop: int) -> list[tuple[int, dict]]:
            return [(i, {"value": i}) for i in range(start, stop)]

        @variable(name="b")
        def gen_b(start: int, stop: int) -> list[tuple[int, dict]]:
            return [(i, {"value": i}) for i in range(start, stop)]

        @experiment()
        def run(a: int, b: int) -> list[dict]:
            return [{"sum": a + b}]

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"
            self._write_config(
                config_path,
                {
                    "a": {"start": 0, "stop": 2},
                    "b": {"start": 0, "stop": 3},
                },
            )
            argv = ["prog", "--config", str(config_path), "--output", str(output_path)]
            with patch.object(sys, "argv", argv):
                run()

            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["a"], 0)
            self.assertEqual(rows[0]["b"], 0)
            self.assertEqual(rows[1]["a"], 0)
            self.assertEqual(rows[1]["b"], 1)
            self.assertIn("a__value", rows[0])
            self.assertIn("b__value", rows[0])

    def test_signature_filtering_and_dict_result(self) -> None:
        @variable(name="a")
        def gen_a(start: int, stop: int) -> list[tuple[int, dict]]:
            return [(i, {}) for i in range(start, stop)]

        @experiment()
        def run(a: int) -> dict:
            return {"value": a}

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"
            self._write_config(
                config_path,
                {
                    "a": {"start": 0, "stop": 2, "step": 1},
                },
            )
            argv = ["prog", "--config", str(config_path), "--output", str(output_path)]
            with patch.object(sys, "argv", argv):
                run()

            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(rows[0]["value"], 0)

    def test_csv_writer_inference(self) -> None:
        @variable(name="a")
        def gen_a(start: int, stop: int) -> list[tuple[int, dict]]:
            return [(i, {}) for i in range(start, stop)]

        @experiment()
        def run(a: int) -> dict:
            return {"value": a}

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.csv"
            self._write_config(config_path, {"a": {"start": 0, "stop": 2}})
            argv = ["prog", "--config", str(config_path), "--output", str(output_path)]
            with patch.object(sys, "argv", argv):
                run()

            header = output_path.read_text().splitlines()[0]
            self.assertIn("a", header)
            self.assertIn("value", header)

    def test_parquet_requires_optional_deps(self) -> None:
        try:
            import pandas  # noqa: F401
        except ImportError:
            pandas = None

        if pandas is not None:
            self.skipTest("pandas installed; parquet behavior depends on pyarrow")

        @variable(name="a")
        def gen_a(start: int, stop: int) -> list[tuple[int, dict]]:
            return [(i, {}) for i in range(start, stop)]

        @experiment()
        def run(a: int) -> dict:
            return {"value": a}

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.parquet"
            self._write_config(config_path, {"a": {"start": 0, "stop": 2}})
            argv = ["prog", "--config", str(config_path), "--output", str(output_path)]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    run()


if __name__ == "__main__":
    unittest.main()
