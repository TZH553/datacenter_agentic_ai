import multiprocessing as mp

import pandas as pd

from controllers import (
    FixedOptimisationController,
    RLController,
    RuleBasedController,
)
from environment import generate_environment
from simulation import run_simulation


HOURS = 24


def calculate_summary(name, df):
    is_agentic = name == "CrewAI Agentic EMS"

    return {
        "System": name,
        "Total Energy (kWh)": df["total_energy_kwh"].sum(),
        "IT Energy (kWh)": df["IT_energy_kwh"].sum(),
        "Cooling Energy (kWh)": df["cooling_energy_kwh"].sum(),
        "Grid Energy (kWh)": df["grid_energy_kwh"].sum(),
        "Solar Used (kWh)": df["solar_used_kwh"].sum(),
        "Solar Curtailed (kWh)": df["solar_curtailed_kwh"].sum(),
        "Battery Discharge (kWh)": df["battery_discharge_kwh"].sum(),
        "Battery Charge (kWh)": df["battery_charge_kwh"].sum(),
        "Unmet Workload (fraction-hours)": (
            df["unmet_workload"] * df["timestep_h"]
        ).sum(),
        "Cost": df["cost"].sum(),
        "Peak Total Power (kW)": df["total_power_kw"].max(),
        "Peak Grid Power (kW)": df["grid_power_kw"].max(),
        "Average PUE": df["pue"].mean(),
        "Average Temperature (C)": df["temperature_C"].mean(),
        "Max Temperature (C)": df["temperature_C"].max(),
        "Temperature Violations": df["temperature_violation"].sum(),
        "Minimum SOC": df["battery_SOC"].min(),
        "Maximum SOC": df["battery_SOC"].max(),
        "Final SOC": df["battery_SOC"].iloc[-1],
        "Average Agent Response Time (s)": (
            df["agent_response_time_s"].mean() if is_agentic else None
        ),
        "Max Agent Response Time (s)": (
            df["agent_response_time_s"].max() if is_agentic else None
        ),
        "Agent Timeouts": (
            df["agent_timed_out"].sum() if is_agentic else None
        ),
        "Fallback Uses": (
            df["fallback_used"].sum() if is_agentic else None
        ),
    }


def main():
    workload, solar, price = generate_environment(HOURS)

    systems = [
        ("Rule-Based EMS", RuleBasedController(), False, "rq3_rule_based_hourly.csv"),
        (
            "Fixed Optimisation",
            FixedOptimisationController(),
            False,
            "rq3_fixed_optimisation_hourly.csv",
        ),
        ("RL Controller", RLController(), False, "rq3_rl_hourly.csv"),
        ("CrewAI Agentic EMS", None, True, "rq3_agentic_hourly.csv"),
    ]

    summaries = []
    for name, controller, is_agentic, output_path in systems:
        results = run_simulation(
            controller=controller,
            workload_profile=workload,
            solar_profile=solar,
            price_profile=price,
            hours=HOURS,
            is_agentic=is_agentic,
        )
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        summaries.append(calculate_summary(name, df))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv("rq3_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    mp.freeze_support()
    main()
