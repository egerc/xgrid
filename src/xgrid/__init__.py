try:
    from importlib.metadata import version

    __version__ = version("xgrid")
except Exception:  # pragma: no cover - package metadata unavailable in tests/source
    __version__ = "0.1.0"

__all__ = ["__version__"]


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main

    return cli_main(argv)
