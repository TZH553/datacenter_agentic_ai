import pandas as pd

from environment import generate_environment

from simulation import run_simulation

from controllers import (
    RuleBasedController,
    FixedOptimisationController,
    RLController
)


HOURS = 1


def calculate_summary(
    name,
    df
):

    return {

        "System":
            name,

        "Total Energy (kWh)":
            df[
                "total_energy_kwh"
            ].sum(),

        "IT Energy (kWh)":
            df[
                "IT_power_kw"
            ].sum(),

        "Cooling Energy (kWh)":
            df[
                "cooling_power_kw"
            ].sum(),

        "Grid Energy (kWh)":
            df[
                "grid_energy_kwh"
            ].sum(),

        "Solar Used (kWh)":
            df[
                "solar_used_kwh"
            ].sum(),

        "Solar Curtailed (kWh)":
            df[
                "solar_curtailed_kwh"
            ].sum(),

        "Battery Discharge (kWh)":
            df[
                "battery_discharge_kwh"
            ].sum(),

        "Battery Charge (kWh)":
            df[
                "battery_charge_kwh"
            ].sum(),

        "Cost":
            df[
                "cost"
            ].sum(),

        "Peak Total Power (kW)":
            df[
                "total_power_kw"
            ].max(),

        "Peak Grid Power (kW)":
            df[
                "grid_power_kw"
            ].max(),

        "Average PUE":
            df[
                "pue"
            ].mean(),

        "Average Temperature (C)":
            df[
                "temperature_C"
            ].mean(),

        "Max Temperature (C)":
            df[
                "temperature_C"
            ].max(),

        "Temperature Violations":
            df[
                "temperature_violation"
            ].sum(),

        "Minimum SOC":
            df[
                "battery_SOC"
            ].min(),

        "Maximum SOC":
            df[
                "battery_SOC"
            ].max(),

        "Final SOC":
            df[
                "battery_SOC"
            ].iloc[-1]
    }

def main():

    # ==========================================
    # SAME ENVIRONMENT FOR ALL CONTROLLERS
    # ==========================================

    workload, solar, price = (
        generate_environment(
            HOURS
        )
    )


    # ==========================================
    # RULE-BASED
    # ==========================================

    rule_results = run_simulation(
        controller=RuleBasedController(),
        workload_profile=workload,
        solar_profile=solar,
        price_profile=price,
        hours=HOURS,
        is_agentic=False
    )

    rule_df = pd.DataFrame(
        rule_results
    )


    # ==========================================
    # FIXED OPTIMISATION
    # ==========================================

    opt_results = run_simulation(
        controller=FixedOptimisationController(),
        workload_profile=workload,
        solar_profile=solar,
        price_profile=price,
        hours=HOURS,
        is_agentic=False
    )

    opt_df = pd.DataFrame(
        opt_results
    )


    # ==========================================
    # RL
    # ==========================================

    rl_results = run_simulation(
        controller=RLController(),
        workload_profile=workload,
        solar_profile=solar,
        price_profile=price,
        hours=HOURS,
        is_agentic=False
    )

    rl_df = pd.DataFrame(
        rl_results
    )


    # ==========================================
    # CREWAI
    # ==========================================

    agentic_results = run_simulation(
        controller=None,
        workload_profile=workload,
        solar_profile=solar,
        price_profile=price,
        hours=HOURS,
        is_agentic=True
    )

    agentic_df = pd.DataFrame(
        agentic_results
    )

    # ==========================================
    # SAVE HOURLY RESULTS
    # ==========================================

    rule_df.to_csv(
        "rq3_rule_based_hourly.csv",
        index=False
    )

    opt_df.to_csv(
        "rq3_fixed_optimisation_hourly.csv",
        index=False
    )

    rl_df.to_csv(
        "rq3_rl_hourly.csv",
        index=False
    )

    agentic_df.to_csv(
        "rq3_agentic_hourly.csv",
        index=False
    )

    summaries = [

    calculate_summary(
        "Rule-Based EMS",
        rule_df
    ),

    calculate_summary(
        "Fixed Optimisation",
        opt_df
    ),

    calculate_summary(
        "RL Controller",
        rl_df
    ),

    calculate_summary(
        "CrewAI Agentic EMS",
        agentic_df
    )
    ]


    summary_df = pd.DataFrame(
        summaries
    )

    main_columns = [

    "System",

    "Total Energy (kWh)",

    "Grid Energy (kWh)",

    "Solar Used (kWh)",

    "Battery Discharge (kWh)",

    "Cost",

    "Peak Grid Power (kW)",

    "Average PUE",

    "Temperature Violations",

    "Final SOC"
    ]


    print()
    print("=" * 120)
    print("RQ3 CONTROLLER COMPARISON")
    print("=" * 120)

    print(
        summary_df[
            main_columns
        ].to_string(
            index=False
        )
    )


    summary_df.to_csv(
        "rq3_comparison.csv",
        index=False
    )


if __name__ == "__main__":
    main()