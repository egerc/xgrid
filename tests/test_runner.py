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

                @experiment()
                def first(a: int):
                    return {"kind": "first", "value": a}

                @experiment()
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

                @experiment()
                def first(a: int):
                    return {"kind": "first", "value": a}

                @experiment()
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

                @experiment()
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

                @experiment()
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

                @experiment()
                def first(a: int):
                    return {"kind": "first", "value": a}
                """,
            )

            with self.assertRaises(SystemExit) as exc:
                xgrid_main(["run", str(script_path), "--output", str(output_path)])
            self.assertEqual(exc.exception.code, 2)

    def test_variable_config_key_resolves_alternate_config_entry(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"

            self._write_script(
                script_path,
                """
                from xgrid import experiment, variable

                @variable(name="learning_rate", config_key="lr")
                def gen_learning_rate(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment()
                def run(learning_rate: int):
                    return {"learning_rate": learning_rate}
                """,
            )
            self._write_config(config_path, {"lr": {"start": 0, "stop": 2}})

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
            self.assertEqual(rows, [{"learning_rate": 0}, {"learning_rate": 1}])

    def test_variable_rejects_duplicate_effective_config_key(self) -> None:
        @variable(name="alpha", config_key="shared")
        def gen_alpha():
            return [(1, {})]

        self.assertEqual(gen_alpha(), [(1, {})])

        with self.assertRaises(ValueError) as exc:
            @variable(name="beta", config_key="shared")
            def gen_beta():
                return [(2, {})]

        self.assertEqual(
            str(exc.exception),
            "Config key 'shared' is already used by variable 'alpha' "
            "(function 'gen_alpha'); cannot register variable 'beta' "
            "(function 'gen_beta').",
        )

    def test_run_fails_for_duplicate_config_key_in_script(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "experiment.py"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "out.jsonl"

            self._write_script(
                script_path,
                """
                from xgrid import experiment, variable

                @variable(name="alpha", config_key="shared")
                def gen_alpha(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @variable(name="beta", config_key="shared")
                def gen_beta(start: int, stop: int):
                    return [(i, {}) for i in range(start, stop)]

                @experiment()
                def run(alpha: int, beta: int):
                    return {"sum": alpha + beta}
                """,
            )
            self._write_config(
                config_path,
                {
                    "shared": {"start": 0, "stop": 1},
                },
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

            message = str(exc.exception)
            self.assertIn("Failed to import script:", message)
            self.assertIn("Config key 'shared' is already used by variable 'alpha'", message)
            self.assertIn("function 'gen_alpha'", message)
            self.assertIn("function 'gen_beta'", message)

    def test_missing_config_mentions_variable_and_config_key(self) -> None:
        @variable(name="learning_rate", config_key="lr")
        def gen_learning_rate(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        @experiment()
        def run(learning_rate: int):
            return {"learning_rate": learning_rate}

        with self.assertRaises(SystemExit) as exc:
            runner_module.build_rows(run, config={"variables": {}}, show_progress=False)

        self.assertIn(
            "Missing variable configs: learning_rate (config key: lr)",
            str(exc.exception),
        )

    def test_extra_config_warning_uses_effective_config_keys(self) -> None:
        @variable(name="learning_rate", config_key="lr")
        def gen_learning_rate(start: int, stop: int):
            return [(i, {}) for i in range(start, stop)]

        @experiment()
        def run(learning_rate: int):
            return {"learning_rate": learning_rate}

        config = {
            "variables": {
                "lr": {"start": 0, "stop": 1},
                "unexpected": {"start": 0, "stop": 1},
            }
        }

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            rows = runner_module.build_rows(run, config=config, show_progress=False)

        self.assertEqual(rows, [{"learning_rate": 0}])
        self.assertEqual(len(captured), 1)
        self.assertIn("Unknown variables in config: unexpected", str(captured[0].message))

    def test_experiment_variables_argument_is_rejected(self) -> None:
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
                def run(a: int):
                    return {"value": a}
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
            self.assertIn("Failed to import script:", message)
            self.assertIn("unexpected keyword argument 'variables'", message)

    def test_cli_forwards_progress_and_log_level(self) -> None:
        with (
            patch("xgrid.cli.configure_logging") as configure_logging_mock,
            patch("xgrid.cli.run_script") as run_script_mock,
        ):
            code = xgrid_main(
                [
                    "run",
                    "experiment.py",
                    "--config",
                    "config.json",
                    "--output",
                    "out.jsonl",
                    "--progress",
                    "--log-level",
                    "DEBUG",
                ]
            )
        self.assertEqual(code, 0)
        configure_logging_mock.assert_called_once_with("DEBUG")
        run_script_mock.assert_called_once()
        kwargs = run_script_mock.call_args.kwargs
        self.assertEqual(kwargs["show_progress"], True)
        self.assertEqual(kwargs["log_level"], "DEBUG")

    def test_cli_rejects_progress_flag_conflict(self) -> None:
        with self.assertRaises(SystemExit) as exc:
            xgrid_main(
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
        @variable(name="a")
        def gen_a(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        @variable(name="b")
        def gen_b(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        @experiment()
        def run(a: int, b: int):
            return {"sum": a + b}

        config = {
            "variables": {
                "a": {"start": 0, "stop": 2},
                "b": {"start": 0, "stop": 2},
            }
        }

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
                run, config=config, show_progress=True
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

    def test_build_rows_without_progress_skips_total_precount(self) -> None:
        @variable(name="a")
        def gen_a(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        @experiment()
        def run(a: int):
            return {"value": a}

        config = {"variables": {"a": {"start": 0, "stop": 2}}}

        class DummyProgress:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def set_postfix_str(self, _text: str) -> None:
                return None

            def update(self, _value: int) -> None:
                return None

        created: list[DummyProgress] = []

        def make_progress(*_args, **kwargs):
            progress = DummyProgress(**kwargs)
            created.append(progress)
            return progress

        with (
            patch("xgrid.runner._compute_total_iterations") as compute_total_mock,
            patch("xgrid.runner.tqdm", side_effect=make_progress),
        ):
            rows = runner_module.build_rows(
                run, config=config, show_progress=False
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(created), 1)
        self.assertIsNone(created[0].kwargs["total"])
        compute_total_mock.assert_not_called()

    def test_build_rows_zero_length_variable_uses_zero_total(self) -> None:
        @variable(name="a")
        def gen_a(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        @experiment()
        def run(a: int):
            return {"value": a}

        config = {"variables": {"a": {"start": 0, "stop": 0}}}

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
                run, config=config, show_progress=True
            )

        self.assertEqual(rows, [])
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].kwargs["total"], 0)
        self.assertEqual(sum(created[0].updates), 0)

    def test_build_rows_reinvokes_inner_variable_generators(self) -> None:
        call_counts = {"a": 0, "b": 0}

        @variable(name="a")
        def gen_a(stop: int):
            call_counts["a"] += 1
            for i in range(stop):
                yield i, {}

        @variable(name="b")
        def gen_b(stop: int):
            call_counts["b"] += 1
            for i in range(stop):
                yield i, {}

        @experiment()
        def run(a: int, b: int):
            return {"sum": a + b}

        config = {"variables": {"a": {"stop": 2}, "b": {"stop": 3}}}

        rows = runner_module.build_rows(
            run, config=config, show_progress=False
        )

        self.assertEqual(len(rows), 6)
        self.assertEqual(call_counts["a"], 1)
        self.assertEqual(call_counts["b"], 2)

    def test_build_rows_progress_precount_adds_variable_invocations(self) -> None:
        call_counts = {"a": 0, "b": 0}

        @variable(name="a")
        def gen_a(stop: int):
            call_counts["a"] += 1
            for i in range(stop):
                yield i, {}

        @variable(name="b")
        def gen_b(stop: int):
            call_counts["b"] += 1
            for i in range(stop):
                yield i, {}

        @experiment()
        def run(a: int, b: int):
            return {"sum": a + b}

        config = {"variables": {"a": {"stop": 2}, "b": {"stop": 3}}}

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
            rows = runner_module.build_rows(
                run, config=config, show_progress=True
            )

        self.assertEqual(len(rows), 6)
        self.assertEqual(call_counts["a"], 2)
        self.assertEqual(call_counts["b"], 3)

    def test_build_rows_auto_progress_respects_tty(self) -> None:
        @variable(name="a")
        def gen_a(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        @experiment()
        def run(a: int):
            return {"value": a}

        config = {"variables": {"a": {"start": 0, "stop": 1}}}
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

    def test_run_script_logs_lifecycle_messages(self) -> None:
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

                @experiment()
                def run(a: int):
                    return {"value": a}
                """,
            )
            self._write_config(config_path, {"a": {"start": 0, "stop": 2}})

            with self.assertLogs("xgrid.runner", level="INFO") as log_context:
                rows = runner_module.run_script(
                    script_path,
                    config_path=config_path,
                    output_path=output_path,
                    show_progress=False,
                    log_level="INFO",
                )

        self.assertEqual(len(rows), 2)
        self.assertTrue(
            any("Starting run" in message for message in log_context.output),
            msg=log_context.output,
        )
        self.assertTrue(
            any(
                "Initialized lazy variable iteration" in message
                for message in log_context.output
            ),
            msg=log_context.output,
        )
        self.assertTrue(
            any("total_iterations=unknown" in message for message in log_context.output),
            msg=log_context.output,
        )
        self.assertTrue(
            any("Completed run" in message for message in log_context.output),
            msg=log_context.output,
        )

    def test_run_script_logs_known_total_when_progress_enabled(self) -> None:
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

                @experiment()
                def run(a: int):
                    return {"value": a}
                """,
            )
            self._write_config(config_path, {"a": {"start": 0, "stop": 2}})

            class DummyProgress:
                def __init__(self, **_kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return None

                def set_postfix_str(self, _text: str) -> None:
                    return None

                def update(self, _value: int) -> None:
                    return None

            with (
                patch("xgrid.runner.tqdm", return_value=DummyProgress()),
                self.assertLogs("xgrid.runner", level="INFO") as log_context,
            ):
                rows = runner_module.run_script(
                    script_path,
                    config_path=config_path,
                    output_path=output_path,
                    show_progress=True,
                    log_level="INFO",
                )

        self.assertEqual(len(rows), 2)
        self.assertTrue(
            any("total_iterations=2" in message for message in log_context.output),
            msg=log_context.output,
        )

    def test_progress_does_not_change_output_rows(self) -> None:
        @variable(name="a")
        def gen_a(start: int, stop: int):
            return [(i, {"value": i}) for i in range(start, stop)]

        @experiment()
        def run(a: int):
            return [{"value": a}, {"double": a * 2}]

        config = {"variables": {"a": {"start": 0, "stop": 2}}}

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

                @experiment()
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
