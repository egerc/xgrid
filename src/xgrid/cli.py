from __future__ import annotations

import argparse
from pathlib import Path

from .runner import configure_logging, run_script


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        configure_logging(args.log_level)
        script_path = _resolve_script_path(args.script, args.script_flag)
        run_script(
            script_path,
            config_path=Path(args.config),
            output_path=Path(args.output),
            output_format=args.format,
            experiment_name=args.experiment,
            show_progress=args.progress,
            log_level=args.log_level,
        )
        return 0
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
