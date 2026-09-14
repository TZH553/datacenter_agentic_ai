import multiprocessing as mp
import os

import pandas as pd

from controllers import (
    FixedOptimisationController,
    RLController,
    RuleBasedController,
)
from environment import generate_environment
from simulation import run_simulation


# Keep defaults small for an architecture smoke test. Override for final runs:
# Windows CMD: set EXPERIMENT_HOURS=24
# PowerShell:  $env:EXPERIMENT_HOURS=24
HOURS = int(os.getenv("EXPERIMENT_HOURS", "24"))
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "120"))


def calculate_summary(name, df, is_agentic):
    return {
        "System": name,
        "Total Energy (kWh)": df["total_energy_kwh"].sum(),
        "IT Energy (kWh)": df["IT_energy_kwh"].sum(),
        "Cooling Energy (kWh)": df["cooling_energy_kwh"].sum(),
        "Grid Energy (kWh)": df["grid_energy_kwh"].sum(),
        "Solar Used (kWh)": df["solar_used_kwh"].sum(),
        "Solar to Load (kWh)": df["solar_to_load_kwh"].sum(),
        "Solar to Battery (kWh)": df["solar_to_battery_kwh"].sum(),
        "Solar Curtailed (kWh)": df["solar_curtailed_kwh"].sum(),
        "Grid to Load (kWh)": df["grid_to_load_kwh"].sum(),
        "Grid to Battery (kWh)": df["grid_to_battery_kwh"].sum(),
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
        "Prompt Tokens": (
            df["agent_prompt_tokens"].sum() if is_agentic else None
        ),
        "Completion Tokens": (
            df["agent_completion_tokens"].sum() if is_agentic else None
        ),
        "Total Tokens": (
            df["agent_total_tokens"].sum() if is_agentic else None
        ),
    }


def main():
    workload, solar, price = generate_environment(HOURS)

    systems = [
        {
            "name": "Rule-Based EMS",
            "controller": RuleBasedController(),
            "is_agentic": False,
            "architecture": "four_agent",
            "output": "rq3_rule_based_hourly.csv",
        },
        {
            "name": "Fixed Optimisation",
            "controller": FixedOptimisationController(),
            "is_agentic": False,
            "architecture": "four_agent",
            "output": "rq3_fixed_optimisation_hourly.csv",
        },
        {
            "name": "RL Controller",
            "controller": RLController(),
            "is_agentic": False,
            "architecture": "four_agent",
            "output": "rq3_rl_hourly.csv",
        },
        {
            "name": "CrewAI Four-Agent",
            "controller": None,
            "is_agentic": True,
            "architecture": "four_agent",
            "output": "rq3_agentic_four_hourly.csv",
        },
        {
            "name": "CrewAI Three-Agent",
            "controller": None,
            "is_agentic": True,
            "architecture": "three_agent",
            "output": "rq3_agentic_three_hourly.csv",
        },
        {
            "name": "CrewAI Single-Agent",
            "controller": None,
            "is_agentic": True,
            "architecture": "single_agent",
            "output": "rq3_agentic_single_hourly.csv",
        },
    ]

    summaries = []
    for system in systems:
        results = run_simulation(
            controller=system["controller"],
            workload_profile=workload,
            solar_profile=solar,
            price_profile=price,
            hours=HOURS,
            is_agentic=system["is_agentic"],
            agentic_architecture=system["architecture"],
            agent_timeout_seconds=AGENT_TIMEOUT_SECONDS,
        )
        df = pd.DataFrame(results)
        df.to_csv(system["output"], index=False)
        summaries.append(
            calculate_summary(
                system["name"],
                df,
                system["is_agentic"],
            )
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv("rq3_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    mp.freeze_support()
    main()
