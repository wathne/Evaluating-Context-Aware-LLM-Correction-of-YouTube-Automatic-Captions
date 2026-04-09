# Helper functions.


from random import randrange
from time import sleep
from typing import Any


def float_or_none(x: Any, /) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        pass
    return None


def float_or_none_if_zero(x: Any, /) -> float | None:
    if x is None:
        return None
    try:
        1/float(x) # type: ignore
        return float(x)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        pass
    return None


def float_or_zero_if_none(x: Any, /) -> float:
    if x is None:
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        pass
    return 0.0


def int_or_none(x: Any, /) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        pass
    return None


def int_or_none_if_zero(x: Any, /) -> int | None:
    if x is None:
        return None
    try:
        1/int(x) # type: ignore
        return int(x)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        pass
    return None


def int_or_zero_if_none(x: Any, /) -> int:
    if x is None:
        return 0
    try:
        return int(x)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        pass
    return 0


def str_or_none(object: Any) -> str | None:
    if object is None:
        return None
    return str(object)


def str_or_none_if_empty(object: Any) -> str | None:
    if object is None:
        return None
    if str(object) == "":
        return None
    return str(object)


def str_or_empty_if_none(object: Any) -> str:
    if object is None:
        return ""
    return str(object)


def randomized_wait(
    wait_milliseconds_min: int | None = 120000, # 120 seconds
    wait_milliseconds_max: int | None = 240000, # 240 seconds
    verbose: bool = True,
) -> None:
    step: int = 1

    try:
        verbose = bool(verbose)
    except (TypeError, ValueError):
        verbose = True

    wait_milliseconds_min = int_or_none(wait_milliseconds_min)
    if (wait_milliseconds_min is None or
        wait_milliseconds_min < 0
    ):
        wait_milliseconds_min = 0

    wait_milliseconds_max = int_or_none(wait_milliseconds_max)
    if (wait_milliseconds_max is None or
        wait_milliseconds_max < wait_milliseconds_min + step
    ):
        wait_milliseconds_max = wait_milliseconds_min + step

    wait_milliseconds_random: int
    try:
        wait_milliseconds_random = randrange(
            start=wait_milliseconds_min,
            stop=wait_milliseconds_max,
            step=step,
        )
    except (TypeError, ValueError):
        wait_milliseconds_random = wait_milliseconds_min

    wait_seconds: float = float(wait_milliseconds_random / 1000)

    if verbose:
        fractions: int = 4
        fraction_seconds: float = float(wait_seconds / fractions)

        print(f"Waiting {wait_milliseconds_random} ms ", end="", flush=True)
        sleep(fraction_seconds)
        print(".", end="", flush=True)
        sleep(fraction_seconds)
        print(".", end="", flush=True)
        sleep(fraction_seconds)
        print(".", end="", flush=True)
        sleep(fraction_seconds)
        print("done.")

        return None

    sleep(wait_seconds)

    return None


def input_to_continue() -> bool:
    key: str
    while True:
        key = str(input("Input y to continue or n to exit: "))

        if key == "y" or key == "Y":
            return True
        if key == "n" or key == "N":
            return False
