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

    workload: FloatArray = np.asarray(
        np.clip(workload, 0.20, 0.95),
        dtype=np.float64,
    )


    # ---------------------------------
    # Solar
    # ---------------------------------

    solar: FloatArray = np.asarray(
        np.maximum(
            0.0,
            50.0 * np.sin((hour - 6) / 12 * np.pi),
        ),
        dtype=np.float64,
    )


    # ---------------------------------
    # Price
    # ---------------------------------

    price: FloatArray = np.full(
        hours,
        0.20,
        dtype=np.float64,
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


    ambient: FloatArray = np.asarray(
        27.0 + 5.0 * np.sin((hour - 8) / 24 * 2 * np.pi),
        dtype=np.float64,
    )
    if include_ambient:
        return workload, solar, price, ambient
    return workload, solar, price
