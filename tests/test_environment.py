import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xgrid import environment as environment_module


class EnvironmentTests(unittest.TestCase):
    def test_parse_environment_rejects_non_object(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text("{}")
            with self.assertRaises(SystemExit) as exc:
                environment_module.parse_environment_config(
                    {"environment": "invalid"},
                    config_path=config_path,
                )
            self.assertEqual(
                str(exc.exception), "Config 'environment' must be an object"
            )

    def test_select_backend_auto_prefers_config_then_uv_default(self) -> None:
        parsed_with_backend = environment_module.ParsedEnvironmentConfig(
            backend="uv",
            python=None,
            dependencies=(),
            requirements_files=(),
        )
        self.assertEqual(
            environment_module.select_backend(
                "auto",
                parsed_environment=parsed_with_backend,
            ),
            "uv",
        )
        parsed_without_backend = environment_module.ParsedEnvironmentConfig(
            backend=None,
            python=None,
            dependencies=(),
            requirements_files=(),
        )
        self.assertEqual(
            environment_module.select_backend(
                "auto",
                parsed_environment=parsed_without_backend,
            ),
            "uv",
        )

    def test_compute_environment_fingerprint_changes_with_dependencies(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            req_path = tmp_path / "requirements.txt"
            req_path.write_text("numpy==2.2.0\n")
            spec_one = environment_module.EnvironmentSpec(
                backend="uv",
                python="3.11",
                dependencies=("pandas==2.2.3",),
                requirements_files=(req_path,),
            )
            fingerprint_one = environment_module.compute_environment_fingerprint(
                spec=spec_one,
            )
            spec_two = environment_module.EnvironmentSpec(
                backend="uv",
                python="3.11",
                dependencies=("pandas==2.2.4",),
                requirements_files=(req_path,),
            )
            fingerprint_two = environment_module.compute_environment_fingerprint(
                spec=spec_two,
            )
            self.assertNotEqual(fingerprint_one, fingerprint_two)

    def test_compute_environment_fingerprint_stable_for_identical_spec(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            req_path = tmp_path / "requirements.txt"
            req_path.write_text("numpy==2.2.0\n")
            spec = environment_module.EnvironmentSpec(
                backend="uv",
                python="3.11",
                dependencies=("pandas==2.2.3",),
                requirements_files=(req_path,),
            )
            fingerprint_one = environment_module.compute_environment_fingerprint(
                spec=spec
            )
            fingerprint_two = environment_module.compute_environment_fingerprint(
                spec=spec
            )
            self.assertEqual(fingerprint_one, fingerprint_two)

    def test_compute_environment_fingerprint_changes_with_requirements_content(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            req_path = tmp_path / "requirements.txt"
            req_path.write_text("numpy==2.2.0\n")
            spec = environment_module.EnvironmentSpec(
                backend="uv",
                python="3.11",
                dependencies=("pandas==2.2.3",),
                requirements_files=(req_path,),
            )
            fingerprint_one = environment_module.compute_environment_fingerprint(
                spec=spec
            )
            req_path.write_text("numpy==2.2.1\n")
            fingerprint_two = environment_module.compute_environment_fingerprint(
                spec=spec
            )
            self.assertNotEqual(fingerprint_one, fingerprint_two)

    def test_compute_environment_fingerprint_changes_with_python_target(self) -> None:
        spec_one = environment_module.EnvironmentSpec(
            python="3.11",
            backend="uv",
            dependencies=("pandas==2.2.3",),
            requirements_files=(),
        )
        spec_two = environment_module.EnvironmentSpec(
            python="3.11",
            backend="uv",
            dependencies=("pandas==2.2.3",),
            requirements_files=(),
        )
        fingerprint_one = environment_module.compute_environment_fingerprint(
            spec=spec_one
        )
        spec_three = environment_module.EnvironmentSpec(
            backend="uv",
            python="3.12",
            dependencies=("pandas==2.2.3",),
            requirements_files=(),
        )
        fingerprint_two = environment_module.compute_environment_fingerprint(
            spec=spec_two
        )
        fingerprint_three = environment_module.compute_environment_fingerprint(
            spec=spec_three
        )
        self.assertEqual(fingerprint_one, fingerprint_two)
        self.assertNotEqual(fingerprint_one, fingerprint_three)

    def test_materialize_lock_collects_inline_and_file_dependencies(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_root = tmp_path / "project"
            cache_dir = tmp_path / "cache"
            project_root.mkdir()
            requirements_path = tmp_path / "requirements.txt"
            requirements_path.write_text("# comment\nnumpy==2.2.0\n")
            spec = environment_module.EnvironmentSpec(
                backend="uv",
                python="3.11",
                dependencies=("pandas==2.2.3",),
                requirements_files=(requirements_path,),
            )
            lock = environment_module.materialize_lock(
                project_root=project_root,
                cache_dir=cache_dir,
                spec=spec,
                refresh_lock=True,
            )
            lock_lines = lock.content.splitlines()
            self.assertIn("pandas==2.2.3", lock_lines)
            self.assertIn("numpy==2.2.0", lock_lines)

    def test_prepare_uv_reports_bootstrap_then_reuse(self) -> None:
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            spec = environment_module.EnvironmentSpec(
                backend="uv",
                python="3.11",
                dependencies=(),
                requirements_files=(),
            )
            with (
                patch(
                    "xgrid.environment.backends.uv.shutil.which",
                    return_value="/usr/bin/uv",
                ),
                patch("xgrid.environment.backends.uv.run_command") as run_command,
            ):
                first = environment_module.prepare_environment(
                    project_root=project_root,
                    spec=spec,
                    fingerprint="abc123",
                    rebuild=False,
                    refresh_lock=False,
                )
                self.assertEqual(first.status, "bootstrap")
                self.assertGreaterEqual(run_command.call_count, 2)
                self.assertIsNotNone(first.python_executable)

                assert first.python_executable is not None
                first.python_executable.parent.mkdir(parents=True, exist_ok=True)
                first.python_executable.write_text("")

                run_command.reset_mock()
                second = environment_module.prepare_environment(
                    project_root=project_root,
                    spec=spec,
                    fingerprint="abc123",
                    rebuild=False,
                    refresh_lock=False,
                )
                self.assertEqual(second.status, "reuse")
                run_command.assert_not_called()

    def test_parse_environment_rejects_docker_backend(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text("{}")
            with self.assertRaises(SystemExit) as exc:
                environment_module.parse_environment_config(
                    {"environment": {"backend": "docker"}},
                    config_path=config_path,
                )
            self.assertEqual(
                str(exc.exception),
                "Docker backend was removed; use 'uv' or omit environment.backend.",
            )

    def test_parse_environment_rejects_environment_docker_block(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text("{}")
            with self.assertRaises(SystemExit) as exc:
                environment_module.parse_environment_config(
                    {"environment": {"docker": {"base_image": "python:3.11-slim"}}},
                    config_path=config_path,
                )
            self.assertEqual(
                str(exc.exception),
                "Config 'environment.docker' is not supported (Docker backend was removed)",
            )


if __name__ == "__main__":
    unittest.main()
