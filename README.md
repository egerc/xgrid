# xgrid

## What xgrid does

xgrid runs parameterized experiments by expanding registered variables into a grid of combinations, executing an experiment function for each combination, and writing tabular results to CSV, JSONL, or Parquet.

## Prerequisites

- Python 3.11+
- `uv` (recommended) or `pip`

## Installation

### Option A (Recommended): uv

```bash
uv sync --dev
```

### Option B: pip editable install

```bash
pip install -e .
```

Use this option if you prefer a traditional Python workflow. Development tools such as `pytest`, `ruff`, and `pyright` are optional unless you are developing the project.

## Quick Start (Showcase)

1. Create a local config file:

```bash
cp config.example.json config.json
```

2. Run the showcase and write CSV output:

```bash
uv run xgrid run showcase.py --config config.json --output results.csv
```

3. Run the same showcase and write JSONL output:

```bash
uv run xgrid run showcase.py --config config.json --output results.jsonl
```

Outputs are written to the path passed to `--output` (for example, `results.csv` or `results.jsonl` in the current directory).

## Writing an Experiment Script

Use `@variable(name="...")` to register variable generators and `@experiment()` to register the experiment function.

```python
from xgrid import experiment, variable


@variable(name="a")
def gen_a(start: int, stop: int, step: int = 1):
    for i in range(start, stop, step):
        yield i, {"value": i}


@variable(name="b")
def gen_b(start: int, stop: int):
    for i in range(start, stop):
        yield i, {"value": i}


@experiment()
def run(a: int, b: int):
    return {"sum": a + b}
```

Notes:
- Each variable generator must yield `(value, metadata_dict)`.
- Experiment functions must return either a `dict` or a `list[dict]`.
- Variable metadata is added to each output row using `<variable_name>__<metadata_key>` (for example, `a__value`).
- Raw variable values are not automatically included in output rows unless your experiment return value includes them.

## Config File Format

Config files are JSON objects with a top-level `variables` object. Each variable entry maps to keyword arguments passed into that variable generator.

```json
{
  "variables": {
    "a": { "start": 0, "stop": 3, "step": 1 },
    "b": { "start": 0, "stop": 2 }
  }
}
```

Behavior:
- Missing required variable configs cause the run to fail.
- Extra variables in config are allowed, but produce warnings.

## CLI Usage

Canonical form:

```bash
xgrid run <script.py> --config <config.json> --output <output_file>
```

You can pass the script path either:
- Positionally: `xgrid run script.py --config config.json --output results.csv`
- With `--script`: `xgrid run --script script.py --config config.json --output results.csv`

Do not provide both forms at once.

Useful flags:
- `--format {csv,jsonl,parquet}` to force output format instead of inferring from extension.
- `--experiment <name>` when the script defines multiple experiments.
- `--log-level {DEBUG,INFO,WARNING,ERROR}` to control runtime logging verbosity (default: `INFO`).
- `--progress` to force-enable the in-place progress bar.
- `--no-progress` to disable the in-place progress bar.

Progress behavior:
- If neither `--progress` nor `--no-progress` is provided, xgrid enables progress only in interactive TTY sessions.
- The progress bar shows current grid iteration metadata in place (for example `a__value=1, b__value=2`).

## Output Formats

xgrid infers output format from `--output` extension:
- `.csv` -> CSV
- `.jsonl` -> JSONL
- `.parquet` -> Parquet

If extension-based inference is not possible, pass `--format`.

Parquet output requires `polars` to be available at runtime.

## Running Multiple Experiments

If your script registers more than one experiment, you must select one with `--experiment`.

```bash
xgrid run multi.py --config config.json --output out.jsonl --experiment second
```

Without `--experiment`, the CLI exits and lists available experiment names.

## Common Errors and Fixes

- Missing script path
  - Cause: omitted both positional script and `--script`.
  - Fix: provide exactly one script path form.
- Missing config or output arguments
  - Cause: required CLI flags omitted.
  - Fix: add `--config <file>` and `--output <file>`.
- No registered variables or experiments
  - Cause: script did not apply `@variable` or `@experiment`.
  - Fix: ensure decorators are present and executed during import.
- Multiple experiments without `--experiment`
  - Cause: script defines multiple `@experiment` functions.
  - Fix: pass `--experiment <name>`.
- Unknown experiment name
  - Cause: value passed to `--experiment` does not exist.
  - Fix: use one of the names listed in the error output.
- Invalid config JSON or wrong schema
  - Cause: malformed JSON or missing top-level `variables` object.
  - Fix: validate JSON and match the documented schema.
- Unsupported or uninferable output format
  - Cause: unsupported extension and no valid `--format`.
  - Fix: use `.csv`, `.jsonl`, `.parquet`, or pass `--format`.

## Development Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run pyright
```

## Pre-commit Hooks

Install and enable the repository-managed pre-commit hook:

```bash
uv sync --dev
uv run pre-commit install
```

Run all hooks manually at any time:

```bash
uv run pre-commit run --all-files
```

The pre-commit hook runs `ruff check .`, `pyright`, and the full `pytest` suite on each commit, stopping on the first failure.
