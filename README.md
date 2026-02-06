# xgrid

## Quick start

1) Create a config:

```
cp config.example.json config.json
```

2) Run the showcase:

```
xgrid run showcase.py --config config.json --output results.csv
```

Direct calls to `@experiment` functions no longer run the grid CLI.

If you want to import `xgrid` from other scripts without modifying `sys.path`, install in editable mode:

```
pip install -e .
```
