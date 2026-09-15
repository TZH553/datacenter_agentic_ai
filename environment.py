import numpy as np
from numpy.typing import NDArray
from typing import Literal, overload


FloatArray = NDArray[np.float64]


@overload
def generate_environment(
    hours: int = 24,
    include_ambient: Literal[False] = False,
) -> tuple[FloatArray, FloatArray, FloatArray]: ...


@overload
def generate_environment(
    hours: int,
    include_ambient: Literal[True],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]: ...


def generate_environment(
    hours: int = 24,
    include_ambient: bool = False,
) -> (
    tuple[FloatArray, FloatArray, FloatArray]
    | tuple[FloatArray, FloatArray, FloatArray, FloatArray]
):

    hour = np.arange(hours)


    # ---------------------------------
    # Workload
    # ---------------------------------

    workload = (
        0.55
        +
        0.25
        * np.sin(
            (hour - 8)
            / 24
            * 2
            * np.pi
        )
    )

    workload = np.clip(
        workload,
        0.20,
        0.95
    )


    # ---------------------------------
    # Solar
    # ---------------------------------

    solar = np.maximum(
        0,

        50
        * np.sin(
            (hour - 6)
            / 12
            * np.pi
        )
    )


    # ---------------------------------
    # Price
    # ---------------------------------

    price = np.full(
        hours,
        0.20
    )


    price[
        (hour >= 8)
        &
        (hour < 12)
    ] = 0.30


    price[
        (hour >= 12)
        &
        (hour < 17)
    ] = 0.15


    price[
        (hour >= 17)
        &
        (hour < 22)
    ] = 0.42


    ambient = 27 + 5 * np.sin((hour - 8) / 24 * 2 * np.pi)
    if include_ambient:
        return workload, solar, price, ambient
    return workload, solar, price
