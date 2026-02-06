# Repository Guidelines

## Project Structure & Module Organization
- `src/xgrid/__init__.py`: Public API (`experiment`, `variable`) and package entrypoint.
- `src/xgrid/cli.py`: Argparse CLI for running experiments (`xgrid run ...`).
- `src/xgrid/registry.py`: Variable/experiment decorators and in-memory registries.
- `src/xgrid/runner.py`: Script loading, config validation, grid expansion, and output writers (`csv`, `jsonl`, `parquet`).
- `tests/test_runner.py`: `unittest` test coverage for CLI behavior and runner execution paths.
- `showcase.py`: Example experiment script loaded by the CLI.
- `config.example.json`: Template config; copy to `config.json` for local runs.
- `pyproject.toml` and `uv.lock`: Project metadata and pinned dependency lockfile.
- `README.md`: Quick-start usage for the current CLI flow.

## Build, Test, and Development Commands
This is a pure-Python project; no build step is required.
- `uv sync --dev`
  Installs runtime and dev dependencies from `pyproject.toml`/`uv.lock`.
- `cp config.example.json config.json`
  Creates a local config file for examples.
- `uv run xgrid run showcase.py --config config.json --output results.csv`
  Runs the showcase through the CLI and writes output (format inferred from extension).
- `uv run xgrid run showcase.py --config config.json --output results.jsonl`
  Runs the same experiment and writes JSONL output.
- `uv run pytest`
  Runs the test suite.
- `uv run ruff check .`
  Lints the codebase.
- `uv run ruff format .`
  Formats code (preferred over manual formatting).
- `uv run pyright`
  Runs static type checks.

## Coding Style & Naming Conventions
- Indentation: 4 spaces, PEP 8 style.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep modules small and focused; prefer adding helpers in `src/xgrid/` rather than inline duplication.
- Use Ruff for formatting and linting; keep type hints consistent with Pyright expectations.

## Testing Guidelines
- Tests are written with `unittest` and executed via `pytest`.
- File naming: `tests/test_*.py`.
- Add tests for new behavior or edge cases, especially around CLI argument handling, experiment selection, config parsing, and output formats.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative summaries (e.g., “Add config validation”).
- PRs should include: a concise description, testing notes (commands + results), and any config or output format changes.
- If a change affects the example workflow, update `showcase.py`, `config.example.json`, and `README.md` together.

## Configuration Notes
- Use `config.json` (local copy) to avoid modifying `config.example.json`.
- Keep example configs minimal and aligned with current decorator/CLI expectations.
- Output format is inferred from `--output` extension unless `--format` is provided explicitly.
