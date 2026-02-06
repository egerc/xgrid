from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__
from . import environment as environment_module
from . import repro
from .runner import configure_logging, run_script


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _handle_run_command(args)
    raise SystemExit(f"Unknown command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xgrid")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment script")
    run_parser.add_argument("script", nargs="?", help="Path to experiment script")
    run_parser.add_argument(
        "--script", dest="script_flag", help="Path to experiment script"
    )
    run_parser.add_argument("--config", required=True, help="Path to config JSON file")
    run_parser.add_argument(
        "--output", required=True, help="Output path (.csv, .jsonl, .parquet)"
    )
    run_parser.add_argument(
        "--format", choices=["csv", "jsonl", "parquet"], default=None
    )
    run_parser.add_argument(
        "--experiment",
        help="Experiment function name when script defines multiple experiments",
    )
    run_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    run_parser.add_argument(
        "--env-backend",
        choices=["auto", "none", "uv", "docker"],
        default="auto",
        help="Environment backend selection",
    )
    run_parser.add_argument(
        "--rebuild-env",
        action="store_true",
        help="Force rebuild of managed environment artifacts",
    )
    run_parser.add_argument(
        "--refresh-lock",
        action="store_true",
        help="Recompute lock material before installation/build",
    )
    progress_group = run_parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        dest="progress",
        action="store_const",
        const=True,
        default=None,
        help="Force-enable the iteration progress bar",
    )
    progress_group.add_argument(
        "--no-progress",
        dest="progress",
        action="store_const",
        const=False,
        help="Disable the iteration progress bar",
    )
    run_parser.add_argument(
        "--_in-managed-env",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser


def _resolve_script_path(
    script_positional: str | None, script_flag: str | None
) -> Path:
    if script_positional and script_flag:
        raise SystemExit(
            "Provide the script path either positionally or with --script, not both."
        )
    script = script_positional or script_flag
    if not script:
        raise SystemExit(
            "Missing script path. Provide a positional script argument or --script."
        )
    return Path(script)


def _handle_run_command(args: argparse.Namespace) -> int:
    script_path = _resolve_script_path(args.script, args.script_flag).expanduser().resolve()
    context = repro.RunContext(
        script_path=script_path,
        config_path=Path(args.config).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        output_format=args.format,
        experiment_name=args.experiment,
        show_progress=args.progress,
        log_level=args.log_level,
    )
    if args._in_managed_env:
        configure_logging(args.log_level)
        run_script(
            context.script_path,
            config_path=context.config_path,
            output_path=context.output_path,
            output_format=context.output_format,
            experiment_name=context.experiment_name,
            show_progress=context.show_progress,
            log_level=context.log_level,
        )
        return 0

    config = _load_config(context.config_path)
    parsed_environment = environment_module.parse_environment_config(
        config,
        config_path=context.config_path,
    )
    selected_backend = environment_module.select_backend(
        args.env_backend,
        parsed_environment=parsed_environment,
    )
    normalized_argv = _normalized_run_argv(
        context=context,
        selected_backend=selected_backend,
        rebuild_env=args.rebuild_env,
        refresh_lock=args.refresh_lock,
    )
    return _execute_run_with_environment(
        context=context,
        selected_backend=selected_backend,
        parsed_environment=parsed_environment,
        rebuild_env=args.rebuild_env,
        refresh_lock=args.refresh_lock,
        normalized_argv=normalized_argv,
    )


def _execute_run_with_environment(
    *,
    context: repro.RunContext,
    selected_backend: str,
    parsed_environment: environment_module.ParsedEnvironmentConfig,
    rebuild_env: bool,
    refresh_lock: bool,
    normalized_argv: list[str],
) -> int:
    logger = configure_logging(context.log_level)
    project_root = Path.cwd().resolve()
    spec = environment_module.build_environment_spec(
        selected_backend=selected_backend,
        parsed_environment=parsed_environment,
    )
    fingerprint: str | None = None
    if spec is not None:
        fingerprint = environment_module.compute_environment_fingerprint(spec=spec)

    prepared = environment_module.prepare_environment(
        project_root=project_root,
        spec=spec,
        fingerprint=fingerprint,
        rebuild=rebuild_env,
        refresh_lock=refresh_lock,
    )
    logger.info(
        "Environment ready backend=%s status=%s fingerprint=%s",
        prepared.backend,
        prepared.status,
        prepared.fingerprint or "none",
    )

    if prepared.backend == "none":
        run_script(
            context.script_path,
            config_path=context.config_path,
            output_path=context.output_path,
            output_format=context.output_format,
            experiment_name=context.experiment_name,
            show_progress=context.show_progress,
            log_level=context.log_level,
        )
    else:
        run_arguments = _build_managed_run_arguments(
            context=context,
            backend=prepared.backend,
            project_root=project_root,
        )
        environment_module.run_in_prepared_environment(
            project_root=project_root,
            prepared=prepared,
            run_arguments=run_arguments,
        )

    manifest_path = repro.write_run_manifest(
        context=context,
        selected_backend=selected_backend,
        environment_fingerprint=prepared.fingerprint,
        lock_fingerprint=(
            prepared.lock_material.fingerprint if prepared.lock_material else None
        ),
        lock_material=(prepared.lock_material.content if prepared.lock_material else None),
        python_version=prepared.python_version,
        environment_status=prepared.status,
        xgrid_version=__version__,
        normalized_cli_argv=normalized_argv,
    )
    logger.info("Wrote run manifest path=%s", manifest_path)
    return 0


def _build_managed_run_arguments(
    *,
    context: repro.RunContext,
    backend: str,
    project_root: Path,
) -> list[str]:
    script_path = context.script_path
    config_path = context.config_path
    output_path = context.output_path
    if backend == "docker":
        script_path = environment_module.rewrite_path_for_docker(
            script_path,
            project_root=project_root,
        )
        config_path = environment_module.rewrite_path_for_docker(
            config_path,
            project_root=project_root,
        )
        output_path = environment_module.rewrite_path_for_docker(
            output_path,
            project_root=project_root,
        )

    args = [
        "run",
        str(script_path),
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--log-level",
        context.log_level,
        "--_in-managed-env",
    ]
    if context.output_format:
        args.extend(["--format", context.output_format])
    if context.experiment_name:
        args.extend(["--experiment", context.experiment_name])
    if context.show_progress is True:
        args.append("--progress")
    if context.show_progress is False:
        args.append("--no-progress")
    return args


def _normalized_run_argv(
    *,
    context: repro.RunContext,
    selected_backend: str,
    rebuild_env: bool,
    refresh_lock: bool,
) -> list[str]:
    args = [
        "run",
        str(context.script_path),
        "--config",
        str(context.config_path),
        "--output",
        str(context.output_path),
        "--log-level",
        context.log_level,
        "--env-backend",
        selected_backend,
    ]
    if context.output_format:
        args.extend(["--format", context.output_format])
    if context.experiment_name:
        args.extend(["--experiment", context.experiment_name])
    if context.show_progress is True:
        args.append("--progress")
    if context.show_progress is False:
        args.append("--no-progress")
    if rebuild_env:
        args.append("--rebuild-env")
    if refresh_lock:
        args.append("--refresh-lock")
    return args


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in config: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Config root must be a JSON object: {path}")
    return payload
