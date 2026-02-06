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

from xgrid import experiment, main as xgrid_main, variable
from xgrid import registry as registry_module
from xgrid import runner as runner_module


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        registry_module._clear_registry()

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
        experiment_names: tuple[str, ...] = ("run", "first", "second"),
    ) -> dict[str, dict]:
        resolved_bindings = (
            bindings
            if bindings is not None
            else {
                self._argument_name_for_variable(variable_name): variable_name
                for variable_name in variables.keys()
            }
        )
        experiments = {
            experiment_name: {"bindings": dict(resolved_bindings)}
            for experiment_name in experiment_names
        }
        return {"variables": variables, "experiments": experiments}

    def _write_config(
        self,
        path: Path,
        variables: dict,
        *,
        bindings: dict[str, str] | None = None,
        experiment_names: tuple[str, ...] = ("run", "first", "second"),
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
                from xgrid import experiment, variable

                @variable
                def gen_a(start: int, stop: int):
                    return [(i, {"value": i}) for i in range(start, stop)]

                @variable
                def gen_b(start: int, stop: int):
                    return [(i, {"value": i}) for i in range(start, stop)]

                @experiment
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
                from xgrid import experiment, variable

                @variable
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment
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
                from xgrid import experiment, variable

                @variable
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment
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

                @variable
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment
                def first(a: int):
                    return {"kind": "first", "value": a}

                @experiment
                def second(a: int):
                    return {"kind": "second", "value": a}
                """,
            )
            self._write_config(config_path, {"gen_a": {"start": 0, "stop": 1}})

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

                @variable
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment
                def first(a: int):
                    return {"kind": "first", "value": a}

                @experiment
                def second(a: int):
                    return {"kind": "second", "value": a}
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

                @variable
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment
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

                @variable
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment
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
                from xgrid import experiment, variable

                @variable
                def gen_a(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment
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
                from xgrid import experiment, variable

                @variable
                def gen_learning_rate(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment
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

    def test_variable_rejects_duplicate_function_name(self) -> None:
        @variable
        def gen_alpha():
            return [(1, {})]

        self.assertEqual(gen_alpha(), [(1, {})])

        with self.assertRaises(ValueError) as exc:

            @variable
            def gen_alpha():
                return [(2, {})]

        self.assertEqual(
            str(exc.exception), "Variable 'gen_alpha' is already registered"
        )

    def test_run_fails_for_unknown_bound_generator_in_config(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"

            self._write_script(
                script_path,
                """
                from xgrid import experiment, variable

                @variable
                def gen_alpha(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @variable
                def gen_beta(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment
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
        @variable
        def gen_learning_rate(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        @experiment
        def run(learning_rate: int):
            return {"learning_rate": learning_rate}

        with self.assertRaises(SystemExit) as exc:
            runner_module.build_rows(
                run,
                config={
                    "variables": {},
                    "experiments": {
                        "run": {
                            "bindings": {
                                "learning_rate": "gen_learning_rate",
                            }
                        }
                    },
                },
                show_progress=False,
            )

        self.assertIn(
            "Missing variable configs: gen_learning_rate (bound to argument "
            "'learning_rate' in experiment 'run')",
            str(exc.exception),
        )

    def test_extra_config_warning_uses_variable_names(self) -> None:
        @variable
        def gen_learning_rate(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        @experiment
        def run(learning_rate: int):
            return {"learning_rate": learning_rate}

        config = {
            "variables": {
                "gen_learning_rate": {"start": 0, "stop": 1},
                "unexpected": {"start": 0, "stop": 1},
            },
            "experiments": {
                "run": {
                    "bindings": {
                        "learning_rate": "gen_learning_rate",
                    }
                }
            },
        }

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            rows = runner_module.build_rows(run, config=config, show_progress=False)

        self.assertEqual(rows, [{"learning_rate": 0}])
        self.assertEqual(len(captured), 1)
        self.assertIn(
            "Unknown variables in config: unexpected", str(captured[0].message)
        )

    def test_missing_experiments_object_is_rejected(self) -> None:
        @variable
        def gen_a(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        @experiment
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
        @variable
        def gen_a(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        @experiment
        def run(a: int):
            return {"value": a}

        with self.assertRaises(SystemExit) as exc:
            runner_module.build_rows(
                run,
                config={
                    "variables": {"gen_a": {"start": 0, "stop": 1}},
                    "experiments": {"run": {}},
                },
                show_progress=False,
            )

        self.assertEqual(
            str(exc.exception), "Config must define object 'experiments.run.bindings'"
        )

    def test_missing_required_binding_is_rejected(self) -> None:
        @variable
        def gen_a(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        @experiment
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
        @variable
        def gen_a(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        @experiment
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
        @variable
        def gen_a(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        @experiment
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
        )

        self.assertEqual(rows, [{"sum": 5}, {"sum": 6}])

    def test_duplicate_generator_bindings_expand_independent_axes(self) -> None:
        @variable
        def gen_a(stop: int):
            for i in range(stop):
                yield i, {}

        @experiment
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
        @variable
        def gen_a(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        @variable
        def gen_b(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        @experiment
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
            rows = runner_module.build_rows(run, config=config, show_progress=True)

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
        @variable
        def gen_a(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        @experiment
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
            rows = runner_module.build_rows(run, config=config, show_progress=True)

        self.assertEqual(rows, [])
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].kwargs["total"], 0)
        self.assertEqual(sum(created[0].updates), 0)

    def test_build_rows_auto_progress_respects_tty(self) -> None:
        @variable
        def gen_a(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        @experiment
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
            runner_module.build_rows(run, config=config, show_progress=None)
        with (
            patch("xgrid.runner.tqdm", side_effect=make_progress),
            patch("xgrid.runner.sys.stderr", new=SimpleNamespace(isatty=lambda: True)),
        ):
            runner_module.build_rows(run, config=config, show_progress=None)

        self.assertEqual(disables, [True, False])

    def test_progress_does_not_change_output_rows(self) -> None:
        @variable
        def gen_a(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        @experiment
        def run(a: int):
            return [{"value": a}, {"double": a * 2}]

        config = self._make_config(variables={"gen_a": {"start": 0, "stop": 2}})

        rows_without_progress = runner_module.build_rows(
            run, config=config, show_progress=False
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
                run, config=config, show_progress=True
            )

        self.assertEqual(rows_without_progress, rows_with_progress)

if __name__ == "__main__":
    unittest.main()
