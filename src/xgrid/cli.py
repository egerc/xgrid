from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from . import repro
from .runner import configure_logging, run_script_detailed


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
        "--output",
        required=True,
        help=(
            "Output file/template or directory. File extensions (.csv, .jsonl, .parquet) "
            "write a single file (single experiment). Include '{experiment}' to write "
            "one file per experiment. Otherwise, --output is treated as a directory "
            "and requires --format."
        ),
    )
    run_parser.add_argument(
        "--format", choices=["csv", "jsonl", "parquet"], default=None
    )
    run_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
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
    script_path = (
        _resolve_script_path(args.script, args.script_flag).expanduser().resolve()
    )
    context = repro.RunContext(
        script_path=script_path,
        config_path=Path(args.config).expanduser().resolve(),
        output_template=Path(args.output).expanduser().resolve(),
        output_format=args.format,
        show_progress=args.progress,
        log_level=args.log_level,
    )

    logger = configure_logging(context.log_level)
    report = run_script_detailed(
        context.script_path,
        config_path=context.config_path,
        output_template=context.output_template,
        output_format=context.output_format,
        show_progress=context.show_progress,
        log_level=context.log_level,
    )

    manifest_paths: list[Path] = []
    all_experiments = report.experiment_catalog
    normalized_argv = _normalized_run_argv(context=context)
    for experiment in report.experiments:
        manifest_paths.append(
            repro.write_run_manifest(
                context=context,
                output_path=experiment.output_path,
                output_template=context.output_template,
                experiment_key=experiment.experiment_key,
                experiment_fn=experiment.fn_name,
                experiments=all_experiments,
                xgrid_version=__version__,
                normalized_cli_argv=normalized_argv,
            )
        )
    for manifest_path in manifest_paths:
        logger.info("Wrote run manifest path=%s", manifest_path)
    return 0


def _normalized_run_argv(
    *,
    context: repro.RunContext,
) -> list[str]:
    args = [
        "run",
        str(context.script_path),
        "--config",
        str(context.config_path),
        "--output",
        str(context.output_template),
        "--log-level",
        context.log_level,
    ]
    if context.output_format:
        args.extend(["--format", context.output_format])
    if context.show_progress is True:
        args.append("--progress")
    if context.show_progress is False:
        args.append("--no-progress")
    return args
