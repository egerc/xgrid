from .registry import experiment, variable

__all__ = ["experiment", "variable"]


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main

    return cli_main(argv)
