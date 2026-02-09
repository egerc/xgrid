import json
import sys
import textwrap
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xgrid import main as xgrid_main
from xgrid import runner as runner_module


class RunnerTests(unittest.TestCase):
    def _run_cli(self, args: list[str]) -> int:
        effective_args = list(args)
        if (
            effective_args
            and effective_args[0] == "run"
            and "--env-backend" not in effective_args
            and "--_in-managed-env" not in effective_args
        ):
            effective_args.extend(["--env-backend", "none"])
        return xgrid_main(effective_args)

    def _argument_name_for_variable(self, variable_name: str) -> str:
        if variable_name.startswith("gen_"):
            return variable_name[len("gen_") :]
        if variable_name.startswith("generator_"):
            return variable_name[len("generator_") :]
        return variable_name

    def _make_config(
        self,
        *,
        variables: dict,
        bindings: dict[str, str] | None = None,
        experiment_names: tuple[str, ...] = ("run",),
    ) -> dict[str, dict]:
        typed_variables = {
            variable_key: {"generator": variable_key, **dict(entry)}
            for variable_key, entry in variables.items()
        }
        resolved_bindings = (
            bindings
            if bindings is not None
            else {
                self._argument_name_for_variable(variable_name): variable_name
                for variable_name in typed_variables.keys()
            }
        )
        experiments = {
            experiment_name: {
                "fn": experiment_name,
                "bindings": dict(resolved_bindings),
            }
            for experiment_name in experiment_names
        }
        return {"variables": typed_variables, "experiments": experiments}

    def _write_config(
        self,
        path: Path,
        variables: dict,
        *,
        bindings: dict[str, str] | None = None,
        experiment_names: tuple[str, ...] = ("run",),
    ) -> None:
        path.write_text(
            json.dumps(
                self._make_config(
                    variables=variables,
                    bindings=bindings,
                    experiment_names=experiment_names,
                )
            )
        )

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
                def gen_a(start: int, stop: int):
                    return [(i, {"value": i}) for i in range(start, stop)]

                def gen_b(start: int, stop: int):
                    return [(i, {"value": i}) for i in range(start, stop)]

                def run(a: int, b: int):
                    return [{"sum": a + b}]
                """,
            )
            self._write_config(
                config_path,
                {
                    "gen_a": {"start": 0, "stop": 2},
                    "gen_b": {"start": 0, "stop": 3},
                },
            )

            code = self._run_cli(
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
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                def run(a: int):
                    return {"value": a}
                """,
            )
            self._write_config(
                config_path, {"gen_a": {"start": 0, "stop": 2, "step": 1}}
            )

            code = self._run_cli(
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
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                def run(a: int):
                    return {"value": a}
                """,
            )
            self._write_config(config_path, {"gen_a": {"start": 0, "stop": 2}})

            self._run_cli(
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

    def test_multiple_experiments_write_outputs_to_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_dir = tmp_path / "out"

            self._write_script(
                script_path,
                """
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                def first(a: int):
                    return {"kind": "first", "value": a}

                def second(a: int):
                    return {"kind": "second", "value": a}
                """,
            )
            self._write_config(
                config_path,
                {"gen_a": {"start": 0, "stop": 1}},
                experiment_names=("first", "second"),
            )

            code = self._run_cli(
                [
                    "run",
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_dir),
                    "--format",
                    "jsonl",
                ]
            )
            self.assertEqual(code, 0)

            first_out = output_dir / "first.jsonl"
            second_out = output_dir / "second.jsonl"
            first_rows = [
                json.loads(line) for line in first_out.read_text().splitlines()
            ]
            second_rows = [
                json.loads(line) for line in second_out.read_text().splitlines()
            ]
            self.assertEqual(len(first_rows), 1)
            self.assertEqual(len(second_rows), 1)
            self.assertTrue(all(row["kind"] == "first" for row in first_rows))
            self.assertTrue(all(row["kind"] == "second" for row in second_rows))

    def test_single_experiment_writes_output_to_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_dir = tmp_path / "out"

            self._write_script(
                script_path,
                """
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                def run(a: int):
                    return {"value": a}
                """,
            )
            self._write_config(config_path, {"gen_a": {"start": 0, "stop": 2}})

            code = self._run_cli(
                [
                    "run",
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_dir),
                    "--format",
                    "jsonl",
                ]
            )
            self.assertEqual(code, 0)

            out_path = output_dir / "run.jsonl"
            rows = [json.loads(line) for line in out_path.read_text().splitlines()]
            self.assertEqual(rows[0]["value"], 0)
            self.assertEqual(rows[1]["value"], 1)

    def test_multiple_experiments_write_outputs_from_template(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_template = tmp_path / "out_{experiment}.jsonl"

            self._write_script(
                script_path,
                """
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                def first(a: int):
                    return {"kind": "first", "value": a}

                def second(a: int):
                    return {"kind": "second", "value": a}
                """,
            )
            self._write_config(
                config_path,
                {"gen_a": {"start": 0, "stop": 2}},
                experiment_names=("first", "second"),
            )

            code = self._run_cli(
                [
                    "run",
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_template),
                ]
            )
            self.assertEqual(code, 0)

            first_out = tmp_path / "out_first.jsonl"
            second_out = tmp_path / "out_second.jsonl"
            first_rows = [
                json.loads(line) for line in first_out.read_text().splitlines()
            ]
            second_rows = [
                json.loads(line) for line in second_out.read_text().splitlines()
            ]
            self.assertEqual(len(first_rows), 2)
            self.assertEqual(len(second_rows), 2)
            self.assertTrue(all(row["kind"] == "first" for row in first_rows))
            self.assertTrue(all(row["kind"] == "second" for row in second_rows))

    def test_rejects_both_script_forms(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"

            self._write_script(
                script_path,
                """
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                def first(a: int):
                    return {"kind": "first", "value": a}
                """,
            )
            self._write_config(config_path, {"gen_a": {"start": 0, "stop": 1}})

            with self.assertRaises(SystemExit) as exc:
                self._run_cli(
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
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                def first(a: int):
                    return {"kind": "first", "value": a}
                """,
            )

            with self.assertRaises(SystemExit) as exc:
                self._run_cli(["run", str(script_path), "--output", str(output_path)])
            self.assertEqual(exc.exception.code, 2)

    def test_explicit_bindings_resolve_argument_to_generator(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"

            self._write_script(
                script_path,
                """
                def gen_learning_rate(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                def run(learning_rate: int):
                    return {"learning_rate": learning_rate}
                """,
            )
            self._write_config(
                config_path,
                {"gen_learning_rate": {"start": 0, "stop": 2}},
                bindings={"learning_rate": "gen_learning_rate"},
                experiment_names=("run",),
            )

            code = self._run_cli(
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
            self.assertEqual(rows, [{"learning_rate": 0}, {"learning_rate": 1}])

    def test_run_fails_for_unknown_bound_generator_in_config(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"

            self._write_script(
                script_path,
                """
                def gen_alpha(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                def gen_beta(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                def run(alpha: int, beta: int):
                    return {"sum": alpha + beta}
                """,
            )
            self._write_config(
                config_path,
                {
                    "gen_alpha": {"start": 0, "stop": 1},
                },
                bindings={"alpha": "gen_alpha", "beta": "missing_generator"},
                experiment_names=("run",),
            )

            with self.assertRaises(SystemExit) as exc:
                self._run_cli(
                    [
                        "run",
                        str(script_path),
                        "--config",
                        str(config_path),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertIn(
                "Unknown variable 'missing_generator' bound to argument 'beta' for "
                "experiment 'run'",
                str(exc.exception),
            )

    def test_missing_bound_variable_config_mentions_argument_and_variable(self) -> None:
        def gen_learning_rate(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        def run(learning_rate: int):
            return {"learning_rate": learning_rate}

        with self.assertRaises(SystemExit) as exc:
            runner_module.build_rows(
                run,
                config={
                    "variables": {},
                    "experiments": {
                        "run": {
                            "fn": "run",
                            "bindings": {
                                "learning_rate": "gen_learning_rate",
                            },
                        }
                    },
                },
                show_progress=False,
            )

        self.assertIn(
            "Unknown variable 'gen_learning_rate' bound to argument 'learning_rate' "
            "for experiment 'run'",
            str(exc.exception),
        )

    def test_extra_config_warning_uses_variable_names(self) -> None:
        def gen_learning_rate(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        def run(learning_rate: int):
            return {"learning_rate": learning_rate}

        config = {
            "variables": {
                "gen_learning_rate": {
                    "generator": "gen_learning_rate",
                    "start": 0,
                    "stop": 1,
                },
                "unexpected": {"generator": "gen_unexpected", "start": 0, "stop": 1},
            },
            "experiments": {
                "run": {
                    "fn": "run",
                    "bindings": {
                        "learning_rate": "gen_learning_rate",
                    },
                }
            },
        }

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            rows = runner_module.build_rows(
                run,
                config=config,
                show_progress=False,
                module=SimpleNamespace(gen_learning_rate=gen_learning_rate),
            )

        self.assertEqual(rows, [{"learning_rate": 0}])
        self.assertEqual(len(captured), 1)
        self.assertIn(
            "Unknown variables in config: unexpected", str(captured[0].message)
        )

    def test_missing_experiments_object_is_rejected(self) -> None:
        def gen_a(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        def run(a: int):
            return {"value": a}

        with self.assertRaises(SystemExit) as exc:
            runner_module.build_rows(
                run,
                config={"variables": {"gen_a": {"start": 0, "stop": 1}}},
                show_progress=False,
            )

        self.assertEqual(
            str(exc.exception), "Config must contain an 'experiments' object"
        )

    def test_missing_bindings_object_is_rejected(self) -> None:
        def gen_a(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        def run(a: int):
            return {"value": a}

        with self.assertRaises(SystemExit) as exc:
            runner_module.build_rows(
                run,
                config={
                    "variables": {
                        "gen_a": {"generator": "gen_a", "start": 0, "stop": 1}
                    },
                    "experiments": {"run": {"fn": "run"}},
                },
                show_progress=False,
            )

        self.assertEqual(
            str(exc.exception), "Config must define object 'experiments.run.bindings'"
        )

    def test_missing_required_binding_is_rejected(self) -> None:
        def gen_a(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        def run(a: int, b: int):
            return {"sum": a + b}

        with self.assertRaises(SystemExit) as exc:
            runner_module.build_rows(
                run,
                config=self._make_config(
                    variables={"gen_a": {"start": 0, "stop": 1}},
                    bindings={"a": "gen_a"},
                    experiment_names=("run",),
                ),
                show_progress=False,
            )

        self.assertEqual(str(exc.exception), "Missing bindings for experiment 'run': b")

    def test_unknown_binding_key_is_rejected(self) -> None:
        def gen_a(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        def run(a: int):
            return {"value": a}

        with self.assertRaises(SystemExit) as exc:
            runner_module.build_rows(
                run,
                config=self._make_config(
                    variables={"gen_a": {"start": 0, "stop": 1}},
                    bindings={"a": "gen_a", "unknown": "gen_a"},
                    experiment_names=("run",),
                ),
                show_progress=False,
            )

        self.assertEqual(
            str(exc.exception), "Unknown bindings for experiment 'run': unknown"
        )

    def test_optional_parameter_can_be_unbound(self) -> None:
        def gen_a(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        def run(a: int, b: int = 5):
            return {"sum": a + b}

        rows = runner_module.build_rows(
            run,
            config=self._make_config(
                variables={"gen_a": {"start": 0, "stop": 2}},
                bindings={"a": "gen_a"},
                experiment_names=("run",),
            ),
            show_progress=False,
            module=SimpleNamespace(gen_a=gen_a),
        )

        self.assertEqual(rows, [{"sum": 5}, {"sum": 6}])

    def test_duplicate_generator_bindings_expand_independent_axes(self) -> None:
        def gen_a(stop: int):
            for i in range(stop):
                yield i, {}

        def run(a: int, b: int):
            return {"a": a, "b": b}

        rows = runner_module.build_rows(
            run,
            config=self._make_config(
                variables={"gen_a": {"stop": 2}},
                bindings={"a": "gen_a", "b": "gen_a"},
                experiment_names=("run",),
            ),
            show_progress=False,
            module=SimpleNamespace(gen_a=gen_a),
        )

        pairs = {(row["a"], row["b"]) for row in rows}
        self.assertEqual(pairs, {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_cli_rejects_progress_flag_conflict(self) -> None:
        with self.assertRaises(SystemExit) as exc:
            self._run_cli(
                [
                    "run",
                    "experiment.py",
                    "--config",
                    "config.json",
                    "--output",
                    "out.jsonl",
                    "--progress",
                    "--no-progress",
                ]
            )
        self.assertEqual(exc.exception.code, 2)

    def test_build_rows_progress_updates_with_metadata(self) -> None:
        def gen_a(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        def gen_b(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        def run(a: int, b: int):
            return {"sum": a + b}

        config = self._make_config(
            variables={
                "gen_a": {"start": 0, "stop": 2},
                "gen_b": {"start": 0, "stop": 2},
            }
        )

        class DummyProgress:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.updates: list[int] = []
                self.postfixes: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def set_postfix_str(self, text: str) -> None:
                self.postfixes.append(text)

            def update(self, value: int) -> None:
                self.updates.append(value)

        created: list[DummyProgress] = []

        def make_progress(*_args, **kwargs):
            progress = DummyProgress(**kwargs)
            created.append(progress)
            return progress

        with patch("xgrid.runner.tqdm", side_effect=make_progress):
            rows = runner_module.build_rows(
                run,
                config=config,
                show_progress=True,
                module=SimpleNamespace(gen_a=gen_a, gen_b=gen_b),
            )

        self.assertEqual(len(rows), 4)
        self.assertEqual(len(created), 1)
        progress = created[0]
        self.assertEqual(progress.kwargs["total"], 4)
        self.assertEqual(progress.kwargs["disable"], False)
        self.assertEqual(sum(progress.updates), 4)
        self.assertEqual(len(progress.postfixes), 4)
        self.assertIn("a__value=0", progress.postfixes[0])
        self.assertIn("b__value=0", progress.postfixes[0])

    def test_build_rows_zero_length_variable_uses_zero_total(self) -> None:
        def gen_a(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        def run(a: int):
            return {"value": a}

        config = self._make_config(variables={"gen_a": {"start": 0, "stop": 0}})

        class DummyProgress:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.updates: list[int] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def set_postfix_str(self, _text: str) -> None:
                return None

            def update(self, value: int) -> None:
                self.updates.append(value)

        created: list[DummyProgress] = []

        def make_progress(*_args, **kwargs):
            progress = DummyProgress(**kwargs)
            created.append(progress)
            return progress

        with patch("xgrid.runner.tqdm", side_effect=make_progress):
            rows = runner_module.build_rows(
                run,
                config=config,
                show_progress=True,
                module=SimpleNamespace(gen_a=gen_a),
            )

        self.assertEqual(rows, [])
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].kwargs["total"], 0)
        self.assertEqual(sum(created[0].updates), 0)

    def test_build_rows_auto_progress_respects_tty(self) -> None:
        def gen_a(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        def run(a: int):
            return {"value": a}

        config = self._make_config(variables={"gen_a": {"start": 0, "stop": 1}})
        disables: list[bool] = []

        class DummyProgress:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def set_postfix_str(self, _text: str) -> None:
                return None

            def update(self, _value: int) -> None:
                return None

        def make_progress(*_args, **kwargs):
            disables.append(kwargs["disable"])
            return DummyProgress()

        with (
            patch("xgrid.runner.tqdm", side_effect=make_progress),
            patch("xgrid.runner.sys.stderr", new=SimpleNamespace(isatty=lambda: False)),
        ):
            runner_module.build_rows(
                run,
                config=config,
                show_progress=None,
                module=SimpleNamespace(gen_a=gen_a),
            )
        with (
            patch("xgrid.runner.tqdm", side_effect=make_progress),
            patch("xgrid.runner.sys.stderr", new=SimpleNamespace(isatty=lambda: True)),
        ):
            runner_module.build_rows(
                run,
                config=config,
                show_progress=None,
                module=SimpleNamespace(gen_a=gen_a),
            )

        self.assertEqual(disables, [True, False])

    def test_progress_does_not_change_output_rows(self) -> None:
        def gen_a(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        def run(a: int):
            return [{"value": a}, {"double": a * 2}]

        config = self._make_config(variables={"gen_a": {"start": 0, "stop": 2}})

        rows_without_progress = runner_module.build_rows(
            run,
            config=config,
            show_progress=False,
            module=SimpleNamespace(gen_a=gen_a),
        )

        class DummyProgress:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def set_postfix_str(self, _text: str) -> None:
                return None

            def update(self, _value: int) -> None:
                return None

        with patch("xgrid.runner.tqdm", return_value=DummyProgress()):
            rows_with_progress = runner_module.build_rows(
                run,
                config=config,
                show_progress=True,
                module=SimpleNamespace(gen_a=gen_a),
            )

        self.assertEqual(rows_without_progress, rows_with_progress)


if __name__ == "__main__":
    unittest.main()
