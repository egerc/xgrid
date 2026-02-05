# Repository Guidelines

## Project Structure & Module Organization
- `src/xgrid/`: Core library code (decorators, registry, runner).
- `tests/`: Unit tests (currently `tests/test_runner.py`).
- `showcase.py`: Example script that runs a small experiment.
- `config.example.json`: Template config; copy to `config.json` for local runs.
- `pyproject.toml`: Project metadata, dependencies, and tooling.

## Build, Test, and Development Commands
This is a pure-Python project; no build step is required.
- `cp config.example.json config.json`
  Creates a local config file for examples.
- `uv run python showcase.py --output results.csv`
  Runs the example and writes output to CSV.
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
- Add tests for new behavior or edge cases, especially around config parsing and output formats.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative summaries (e.g., “Add config validation”).
- PRs should include: a concise description, testing notes (commands + results), and any config or output format changes.
- If a change affects the example workflow, update `showcase.py` and `config.example.json` together.

## Configuration Notes
- Use `config.json` (local copy) to avoid modifying `config.example.json`.
- Keep example configs minimal and aligned with current API expectations.
