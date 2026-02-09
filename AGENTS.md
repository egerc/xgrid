# Repository Guidelines

## Project Structure & Module Organization
- `src/xgrid/__init__.py`: Package `main()` shim and version export.
- `src/xgrid/__main__.py`: Module entrypoint for `python -m xgrid`.
- `src/xgrid/cli.py`: Argparse CLI (`xgrid run ...`) including environment and progress flags.
- `src/xgrid/runner/`: Script loading, config validation, experiment resolution, grid execution, and output writers (`csv`, `jsonl`, `parquet`).
- `src/xgrid/environment/`: Managed environment parsing, backend selection, fingerprinting, lock materialization, and `uv` execution.
- `src/xgrid/repro.py`: Reproducibility sidecar helpers (`<output>.run.json`) and input hashing.
- `tests/test_runner.py`: Core CLI and runner execution behavior.
- `tests/test_environment.py`: Environment parsing/fingerprint/preparation behavior.
- `tests/test_cli_repro.py`: CLI reproducibility sidecar and `python -m xgrid` entrypoint coverage.
- `showcase.py`: Example experiment script loaded by the CLI.
- `config.example.json`: Template config, including optional `environment` block.
- `pyproject.toml` and `uv.lock`: Project metadata and pinned dependency lockfile.
- `README.md`: User-facing usage and workflow details.

## Build, Test, and Development Commands
This is a pure-Python project; no build step is required.
- `uv sync --dev`
  Installs runtime and dev dependencies from `pyproject.toml`/`uv.lock`.
- `cp config.example.json config.json`
  Creates a local config file for examples.
- `uv run xgrid run showcase.py --config config.json --output output.csv`
  Runs the showcase through the CLI and writes output (format inferred from extension).
- `uv run xgrid run showcase.py --config config.json --output output.jsonl`
  Runs the same experiment and writes JSONL output. Each run also writes `<output>.run.json`.
- `uv run xgrid run showcase.py --config config.json --output output.parquet`
  Runs the same experiment and writes Parquet output.
- `uv run xgrid run showcase.py --config config.json --output output.csv --env-backend none`
  Runs without managed environment orchestration.
- `uv run pytest`
  Runs the test suite.
- `uv run ruff check .`
  Lints the codebase.
- `uv run ruff format .`
  Formats code (preferred over manual formatting).
- `uv run pyright`
  Runs static type checks.
- `uv run pre-commit run --all-files`
  Runs the configured pre-commit hooks locally.

## Coding Style & Naming Conventions
- Indentation: 4 spaces, PEP 8 style.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep modules small and focused; prefer adding helpers in `src/xgrid/` rather than inline duplication.
- Use Ruff for formatting and linting; keep type hints consistent with Pyright expectations.

## Testing Guidelines
- Tests are written with `unittest` and executed via `pytest`.
- File naming: `tests/test_*.py`.
- Add tests for new behavior or edge cases, especially around CLI argument handling, experiment selection, config parsing, managed environment behavior, reproducibility sidecars, and output formats.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative summaries (e.g., “Add config validation”).
- PRs should include: a concise description, testing notes (commands + results), and any config or output format changes.
- If a change affects the example workflow, update `showcase.py`, `config.example.json`, and `README.md` together.
- If a change affects CLI flags, managed environments, or sidecar schema, update tests and README in the same PR.

## Configuration Notes
- Use `config.json` (local copy) to avoid modifying `config.example.json`.
- Keep example configs minimal and aligned with current decorator/CLI expectations.
- Optional top-level `environment` config controls managed backend (`uv`) when CLI backend is `auto`.
- Managed environment artifacts are cached under `.xgrid/envs/<fingerprint>/`.
- Each successful run writes a reproducibility sidecar at `<output>.run.json`.
- Output format is inferred from `--output` extension unless `--format` is provided explicitly.
