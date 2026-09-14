import pandas as pd

from environment import generate_environment

from controllers import (
    RuleBasedController
)

from simulation import (
    run_simulation
)

import multiprocessing as mp

HOURS = 24


def main():

    print()
    print("=" * 60)
    print("DATA CENTRE SIMULATION")
    print("=" * 60)


    # ==========================================
    # GENERATE IDENTICAL ENVIRONMENT
    # ==========================================

    workload, solar, price = (
        generate_environment(
            HOURS
        )
    )


    print(
        f"Generated {HOURS} hours of input data."
    )


    # ==========================================
    # CREATE CONTROLLER
    # ==========================================

    controller = (
        RuleBasedController()
    )


    # ==========================================
    # RUN
    # ==========================================

    results = run_simulation(

        controller=controller,

        workload_profile=workload,

        solar_profile=solar,

        price_profile=price,

        hours=HOURS,

        is_agentic=False
    )


    # ==========================================
    # DATAFRAME
    # ==========================================

    df = pd.DataFrame(
        results
    )


    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        df.to_string(
            index=False
        )
    )


    # ==========================================
    # SUMMARY
    # ==========================================

    total_energy = (
        df["grid_energy_kwh"].sum()
    )

    total_cost = (
        df["cost"].sum()
    )

    peak_power = (
        df["grid_power_kw"].max()
    )

    average_pue = (
        df["pue"].mean()
    )

    temperature_violations = (
        df[
            "temperature_violation"
        ].sum()
    )


    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"Total grid energy : "
        f"{total_energy:.2f} kWh"
    )

    print(
        f"Total cost        : "
        f"${total_cost:.2f}"
    )

    print(
        f"Peak grid power   : "
        f"{peak_power:.2f} kW"
    )

    print(
        f"Average PUE       : "
        f"{average_pue:.3f}"
    )

    print(
        f"Temp violations   : "
        f"{temperature_violations}"
    )

    print(
        f"Final battery SOC : "
        f"{df['battery_SOC'].iloc[-1] * 100:.1f}%"
    )


    # ==========================================
    # SAVE
    # ==========================================

    df.to_csv(
        "simulation_results.csv",
        index=False
    )

    print()
    print(
        "Results saved to simulation_results.csv"
    )


# THIS IS REQUIRED
if __name__ == "__main__":
    mp.freeze_support()
    main()