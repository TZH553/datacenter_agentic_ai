import numpy as np


def generate_environment(hours=24):

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


    return (
        workload,
        solar,
        price
    )