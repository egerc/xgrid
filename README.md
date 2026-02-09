# xgrid

## What xgrid does

xgrid runs parameterized experiments by expanding registered variables into a grid of combinations, executing an experiment function for each combination, and writing tabular results to CSV, JSONL, or Parquet.

## Prerequisites

- Python 3.11+
- `uv` (required for managed `uv` environments)
- Optional: Docker (for `--env-backend docker`)

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
uv run xgrid run showcase.py --config config.json --output results_{experiment}.csv
```

3. Run the same showcase and write JSONL output:

```bash
uv run xgrid run showcase.py --config config.json --output results_{experiment}.jsonl
```

Outputs are written to the path passed to `--output` (for example, `results.csv` or `results.jsonl` in the current directory).
Each run also writes a reproducibility sidecar at `<output>.run.json` (for example, `results.csv.run.json`).

## Writing an Experiment Script

```python
def gen_a(start: int, stop: int, step: int = 1):
    for i in range(start, stop, step):
        yield i, {"value": i}


def gen_b(start: int, stop: int):
    for i in range(start, stop):
        yield i, {"value": i}


def run(a: int, b: int):
    return {"sum": a + b}
```

Notes:
- Each variable generator must yield `(value, metadata_dict)`.
- Experiment functions must return either a `dict` or a `list[dict]`.
- Experiments run only variables explicitly bound in config under `experiments.<experiment_name>.bindings`.
- Variable metadata is added to each output row using `<experiment_arg>__<metadata_key>` (for example, `a__value`).
- Raw variable values are not automatically included in output rows unless your experiment return value includes them.

## Config File Format

Config files are JSON objects with top-level `variables` and `experiments` objects.  
They may also include an optional top-level `environment` object.
- `variables.<variable_key>.generator` selects the generator function name in your script module.
- `variables.<variable_key>` additional fields map to keyword arguments passed to that generator.
- `experiments.<experiment_key>.fn` selects the experiment function name in your script module.
- `experiments.<experiment_key>.bindings` maps experiment argument names to variable keys.
- `environment` controls managed runtime setup for environment reuse.

```json
{
  "environment": {
    "backend": "uv",
    "python": "3.11",
    "dependencies": ["numpy==2.2.0"],
    "requirements_files": ["requirements.txt"],
    "docker": {
      "base_image": "python:3.11-slim"
    }
  },
  "variables": {
    "a": { "generator": "gen_a", "start": 0, "stop": 3, "step": 1 },
    "b": { "generator": "gen_b", "start": 0, "stop": 2 }
  },
  "experiments": {
    "main": { "fn": "run", "bindings": { "a": "a", "b": "b" } }
  }
}
```

Behavior:
- Missing required bindings or missing bound variable configs cause the run to fail.
- Extra variables in config are allowed, but produce warnings.
- If `environment` is omitted and `--env-backend auto` is used, xgrid defaults to `uv`.

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
- `--log-level {DEBUG,INFO,WARNING,ERROR}` to control runtime logging verbosity (default: `INFO`).
- `--progress` to force-enable the in-place progress bar.
- `--no-progress` to disable the in-place progress bar.
- `--env-backend {auto,none,uv,docker}` to choose environment orchestration.
- `--rebuild-env` to force rebuild of managed environment artifacts.
- `--refresh-lock` to recompute lock material before installing/building.

Progress behavior:
- If neither `--progress` nor `--no-progress` is provided, xgrid enables progress only in interactive TTY sessions.
- The progress bar shows current grid iteration metadata in place (for example `a__value=1, b__value=2`).
- With `--progress`, xgrid first performs a counting pre-pass to compute an exact total iteration count for a bounded progress bar.
- Variable generators should be finite and deterministic for the same config to keep the computed total accurate.
- Progress-enabled runs invoke variable generators an additional time during the counting pre-pass.

### Managed Environments and Sidecars

- Managed runs use cache path `.xgrid/envs/<fingerprint>/`.
- The fingerprint includes managed environment backend, Python target, dependencies, Docker base image, and requirements file paths with content hashes.
- Each successful run writes `<output>.run.json` with:
  - script/config/output paths
  - script/config hashes
  - selected backend and environment fingerprint
  - lock material and lock fingerprint
  - python version
  - normalized CLI argv

## Output Formats

xgrid infers output format from `--output` extension:
- `.csv` -> CSV
- `.jsonl` -> JSONL
- `.parquet` -> Parquet

If extension-based inference is not possible, pass `--format`.

Parquet output requires `polars` to be available at runtime.

## Running Multiple Experiments

If your config defines multiple experiments under `experiments`, xgrid runs all of them.
In that case, `--output` must include `{experiment}` so each experiment writes to a distinct file.

```bash
xgrid run multi.py --config config.json --output out_{experiment}.jsonl
```

## Common Errors and Fixes

- Missing script path
  - Cause: omitted both positional script and `--script`.
  - Fix: provide exactly one script path form.
- Missing config or output arguments
  - Cause: required CLI flags omitted.
  - Fix: add `--config <file>` and `--output <file>`.
- Multiple experiments but missing `{experiment}` in `--output`
  - Cause: config defines more than one experiment and output path would be overwritten.
  - Fix: use an output template like `results_{experiment}.jsonl`.
- Invalid config JSON or wrong schema
  - Cause: malformed JSON or missing required `variables` / `experiments.<name>.bindings` objects.
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
