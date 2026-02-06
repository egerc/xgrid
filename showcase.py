from typing import Dict, Generator, List, Tuple

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from xgrid import experiment, variable


@variable(name="a")
def generator_a(
    start: int, stop: int, step: int = 1
) -> Generator[Tuple[int, Dict], None, None]:
    for i in range(start, stop, step):
        yield i, {"value": i}


@variable(name="b")
def generator_b(
    start: int, stop: int, step: int = 1
) -> Generator[Tuple[int, Dict], None, None]:
    for i in range(start, stop, step):
        yield i, {"value": i}


@variable(name="c")
def generator_c(
    start: int, stop: int, step: int = 1
) -> Generator[Tuple[int, Dict], None, None]:
    for i in range(start, stop, step):
        yield i, {"value": i}


@experiment()
def my_experiment(a: int, b: int, c: int) -> List[Dict]:
    polynomial = a * b + c
    sum = a + b + c
    product = a * b * c
    return [{"polynomial": polynomial, "sum": sum, "product": product}]
