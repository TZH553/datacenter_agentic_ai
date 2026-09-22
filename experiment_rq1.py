import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd

from config import Config
from controllers import (
    FixedOptimisationController,
    RLController,
    RuleBasedController,
)
from environment import generate_environment
from simulation import run_simulation
from trace_workload import load_cloud_workload_trace


# Keep defaults small for an architecture smoke test. Override for final runs:
# Windows CMD: set EXPERIMENT_HOURS=24
# PowerShell:  $env:EXPERIMENT_HOURS=24
# PowerShell lifecycle sensitivity:
# $env:BATTERY_CYCLE_LIFE=10000  # or 50000
# $env:BATTERY_CAPEX_PER_KWH=400
HOURS = int(os.getenv("EXPERIMENT_HOURS", "24"))
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "120"))
BATTERY_CAPEX_PER_KWH = float(
    os.getenv("BATTERY_CAPEX_PER_KWH", "400")
)
BATTERY_CYCLE_LIFE = int(os.getenv("BATTERY_CYCLE_LIFE", "10000"))


def calculate_summary(name, df, is_agentic):
    return {
        "System": name,
        "Workload Source": df["workload_source"].iloc[0],
        "Trace Window Start": df["trace_bin_start"].iloc[0],
        "Hardware Profile": df["hardware_profile"].iloc[0],
        "Processor Model": df["processor_model"].iloc[0],
        "Accelerator Model": df["accelerator_model"].iloc[0],
        "Processing Unit": df["processing_unit_name"].iloc[0],
        "Servers per Cluster": df["servers_per_cluster"].iloc[0],
        "Total Server Count": df["total_server_count"].iloc[0],
        "Capacity per Server": df[
            "processing_capacity_per_server"
        ].iloc[0],
        "Server Idle Power (kW)": df[
            "server_idle_power_kw"
        ].iloc[0],
        "Server Maximum Power (kW)": df[
            "server_max_power_kw"
        ].iloc[0],
        "GPU Server Count": df["gpu_server_count"].iloc[0],
        "GPUs per GPU Server": df["gpus_per_gpu_server"].iloc[0],
        "Total GPU Capacity": df["total_gpu_capacity"].iloc[0],
        "GPU Idle Power (kW)": df["gpu_idle_power_kw"].iloc[0],
        "GPU Maximum Power (kW)": df["gpu_max_power_kw"].iloc[0],
        "Total Energy (kWh)": df["total_energy_kwh"].sum(),
        "IT Energy (kWh)": df["IT_energy_kwh"].sum(),
        "CPU Server Energy (kWh)": (
            df["CPU_server_power_kw"] * df["timestep_h"]
        ).sum(),
        "Accelerator Energy (kWh)": (
            df["accelerator_power_kw"] * df["timestep_h"]
        ).sum(),
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
        "Battery CAPEX ($/kWh)": df["battery_capex_per_kwh"].iloc[0],
        "Battery Cycle Life": df["battery_cycle_life"].iloc[0],
        "Battery Wear Cost ($/kWh discharged)": df[
            "battery_degradation_cost_per_kwh"
        ].iloc[0],
        "Grid Electricity Cost": df["grid_energy_cost"].sum(),
        "Battery Degradation Cost": df[
            "battery_degradation_cost"
        ].sum(),
        "Counterfactual Grid Cost Without Battery": df[
            "counterfactual_grid_cost_without_battery"
        ].sum(),
        "Battery Net Saving vs Grid": df[
            "battery_net_saving_vs_grid"
        ].sum(),
        "Equivalent Full Cycles": df[
            "battery_equivalent_full_cycles"
        ].iloc[-1],
        "Unmet Workload (fraction-hours)": (
            df["unmet_workload"] * df["timestep_h"]
        ).sum(),
        "Cost": df["cost"].sum(),
        "Total Cost": df["cost"].sum(),
        "Peak Total Power (kW)": df["total_power_kw"].max(),
        "Peak Grid Power (kW)": df["grid_power_kw"].max(),
        # Energy-weighted PUE over the complete experiment horizon.
        "Average PUE": (
            df["total_energy_kwh"].sum()
            / df["IT_energy_kwh"].sum()
            if df["IT_energy_kwh"].sum() > 1e-9
            else float("nan")
        ),
        "Average Temperature (C)": df["temperature_C"].mean(),
        "Max Temperature (C)": df["temperature_C"].max(),
        "Temperature Violations": df["temperature_violation"].sum(),
        "Power Cap Violations": df["power_cap_violation"].sum(),
        "Operating Limit Violations": df[
            "operating_limit_violation"
        ].sum(),
        "Maximum Power Risk Ratio": df["power_risk_ratio"].max(),
        "Batch Deadline Missed CPU-Hours": (
            df["batch_deadline_missed_cpu_units"] * df["timestep_h"]
        ).sum(),
        "Average Batch Backlog CPU": df["batch_backlog_cpu_units"].mean(),
        "Final Batch Backlog CPU": df[
            "batch_backlog_cpu_units"
        ].iloc[-1],
        "GPU Work Scheduled (GPU-hours)": df[
            "scheduled_gpu_hours"
        ].sum(),
        "Peak Requested Average GPUs": df[
            "requested_average_gpus"
        ].max(),
        "Unmet GPU Demand (GPU-hours)": (
            df["unmet_average_gpus"] * df["timestep_h"]
        ).sum(),
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
        "Scheduling AI Skips": (
            df["scheduling_ai_skipped"].sum() if is_agentic else None
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
    config = Config(
        battery_capex_per_kwh=BATTERY_CAPEX_PER_KWH,
        battery_cycle_life=BATTERY_CYCLE_LIFE,
    )
    _, solar, price, ambient = generate_environment(
        HOURS, include_ambient=True
    )
    trace_path = (
        Path(__file__).resolve().parent
        / config.trace_workload_filename
    )
    workload, trace_arrivals = load_cloud_workload_trace(
        trace_path,
        HOURS,
        config,
    )

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
            "name": "CrewAI Two-Agent",
            "controller": None,
            "is_agentic": True,
            "architecture": "two_agent",
            "output": "rq3_agentic_two_hourly.csv",
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
            ambient_profile=ambient,
            hours=HOURS,
            is_agentic=system["is_agentic"],
            agentic_architecture=system["architecture"],
            agent_timeout_seconds=AGENT_TIMEOUT_SECONDS,
            config=config,
            trace_arrivals=trace_arrivals,
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
