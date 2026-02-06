from typing import Dict, Generator, List, Tuple
from time import sleep

from xgrid import experiment, variable


@variable()
def generator_a(
    start: int, stop: int, step: int = 1
) -> Generator[Tuple[int, Dict], None, None]:
    for i in range(start, stop, step):
        yield i, {"value": i}


@variable()
def generator_b(
    start: int, stop: int, step: int = 1
) -> Generator[Tuple[int, Dict], None, None]:
    for i in range(start, stop, step):
        yield i, {"value": i}


@variable()
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
    sleep(0.01)
    return [{"polynomial": polynomial, "sum": sum, "product": product}]
