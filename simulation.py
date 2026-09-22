import copy
import multiprocessing as mp
import queue
import time
import traceback

from control import apply_action
from controllers import RuleBasedController
from models import (
    calculate_cooling_power,
    calculate_cost,
    calculate_grid_power,
    calculate_it_power,
    update_battery,
    update_temperature,
)
from state import reset_state, state
from workload import add_trace_workload, add_workload, advance_batch_queue


def _crew_worker(
    initial_state,
    result_queue,
    architecture,
    skip_compute,
):
    """Execute CrewAI in an isolated process and return its updated state."""
    try:
        state.__dict__.clear()
        state.__dict__.update(copy.deepcopy(initial_state))

        from crew import get_data_center_crew

        data_center_crew = get_data_center_crew(
            architecture,
            skip_compute=skip_compute,
        )
        crew_result = data_center_crew.kickoff()
        raw_usage = getattr(crew_result, "token_usage", None)
        token_usage = {}

        if raw_usage is not None:
            dump_method = getattr(raw_usage, "model_dump", None)
            if callable(dump_method):
                dumped_usage = dump_method()
                if isinstance(dumped_usage, dict):
                    token_usage = dumped_usage
            elif isinstance(raw_usage, dict):
                token_usage = raw_usage
        result_queue.put(
            {
                "status": "success",
                "state": copy.deepcopy(state.__dict__),
                "crew_result": str(crew_result),
                "token_usage": token_usage,
            }
        )
    except Exception:
        result_queue.put(
            {
                "status": "error",
                "error": traceback.format_exc(),
            }
        )


def run_crew_with_timeout(
    architecture,
    timeout_seconds=120,
    skip_compute=False,
):
    """Run CrewAI in a child process that can be stopped on timeout."""
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_crew_worker,
        args=(
            copy.deepcopy(state.__dict__),
            result_queue,
            architecture,
            skip_compute,
        ),
    )
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        print(f"[WARNING] CrewAI exceeded {timeout_seconds} seconds.")
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        result_queue.close()
        return None, True, None, {}

    try:
        message = result_queue.get(timeout=2)
    except queue.Empty:
        result_queue.close()
        return None, False, "CrewAI exited without returning a result.", {}

    result_queue.close()

    if message["status"] == "error":
        return None, False, message["error"], {}

    state.__dict__.clear()
    state.__dict__.update(message["state"])
    return (
        message["crew_result"],
        False,
        None,
        message.get("token_usage", {}),
    )


def run_simulation(
    controller,
    workload_profile,
    solar_profile,
    price_profile,
    hours,
    ambient_profile=None,
    is_agentic=False,
    config=None,
    agentic_architecture="four_agent",
    agent_timeout_seconds=120,
    trace_arrivals=None,
):
    """Run all power and energy calculations using one consistent timestep."""
    reset_state(config)
    results = []
    dt = state.timestep_h

    print()
    print("=" * 60)
    print(
        f"Starting CrewAI {agentic_architecture}"
        if is_agentic
        else f"Starting {controller.name}"
    )
    print("=" * 60)

    if any(len(profile) < hours for profile in (
        workload_profile, solar_profile, price_profile
    )):
        raise ValueError("Every input profile must contain at least 'hours' values.")
    if ambient_profile is not None and len(ambient_profile) < hours:
        raise ValueError("ambient_profile must contain at least 'hours' values.")
    if trace_arrivals is not None and len(trace_arrivals) < hours:
        raise ValueError("trace_arrivals must contain at least 'hours' bins.")

    if is_agentic:
        from crew import ARCHITECTURES
        if agentic_architecture not in ARCHITECTURES:
            raise ValueError(
                f"agentic_architecture must be one of {ARCHITECTURES}"
            )

    for hour in range(hours):
        state.solar_kw = float(solar_profile[hour])
        state.grid_price = float(price_profile[hour])
        if ambient_profile is not None:
            state.ambient_temperature = float(ambient_profile[hour])
        if trace_arrivals is None:
            add_workload(workload_profile[hour])
        else:
            add_trace_workload(trace_arrivals[hour])
        state.battery_command_kw = 0.0

        response_time = 0.0
        timed_out = False
        fallback_used = False
        scheduling_ai_skipped = False
        token_usage = {}

        if is_agentic:
            if trace_arrivals is not None:
                incoming_cpu = float(
                    trace_arrivals[hour].get("arrival_cpu", 0.0)
                )
                scheduling_ai_skipped = (
                    incoming_cpu <= 1e-9
                    and state.batch_backlog_cpu <= 1e-9
                )
            if scheduling_ai_skipped:
                for host in state.hosts.values():
                    host["utilisation"] = 0.0
                    host["active"] = False
                state.batch_served_cpu = 0.0
                state.pending_workload_cpu = 0.0
                print(
                    f"\n[AGENTIC] Hour {hour} has no incoming or queued "
                    "work; skipping AI scheduling."
                )
            print(f"\n[AGENTIC] Running CrewAI for hour {hour}...")
            start_time = time.perf_counter()
            (
                crew_output,
                timed_out,
                crew_error,
                token_usage,
            ) = run_crew_with_timeout(
                architecture=agentic_architecture,
                timeout_seconds=agent_timeout_seconds,
                skip_compute=scheduling_ai_skipped,
            )
            response_time = time.perf_counter() - start_time

            if timed_out or crew_error is not None:
                reason = "timed out" if timed_out else "failed"
                print(f"[FALLBACK] CrewAI {reason}; using rule-based controller.")
                if crew_error:
                    print(crew_error)
                fallback_used = True
                apply_action(RuleBasedController().decide())
            else:
                print(
                    f"[AGENTIC] CrewAI completed in "
                    f"{response_time:.2f} seconds."
                )

                # A rejected/missing compute call must never leave the prior
                # hour's utilisation in place. Tool-level fallback normally
                # prevents this; this boundary check is the final safeguard.
                assigned_cpu = sum(
                    host["utilisation"] * host["cpu_capacity"]
                    for host in state.hosts.values()
                )
                allocation_tolerance = max(
                    1e-6, 0.005 * state.pending_workload_cpu
                )
                if abs(assigned_cpu - state.pending_workload_cpu) > allocation_tolerance:
                    print(
                        "[FALLBACK] Agentic compute plan was incomplete; "
                        "applying the current-hour rule-based plan."
                    )
                    if crew_output:
                        print(f"[AGENT OUTPUT] {crew_output}")
                    fallback_used = True
                    apply_action(RuleBasedController().decide())
        else:
            apply_action(controller.decide())

        # Instantaneous power calculations (kW).
        calculate_it_power()
        calculate_cooling_power()
        state.total_power_kw = state.it_power_kw + state.cooling_power_kw

        # State and supply calculations over dt hours.
        update_temperature(dt)
        update_battery(dt)
        calculate_grid_power()
        calculate_cost(dt)

        charge_power_kw = max(0.0, -state.battery_power_kw)
        discharge_power_kw = max(0.0, state.battery_power_kw)
        solar_used_kw = (
            state.solar_to_load_kw + state.solar_to_battery_kw
        )

        assigned_workload_cpu = sum(
            host["utilisation"] * host["cpu_capacity"]
            for host in state.hosts.values()
        )
        unmet_workload_cpu = max(
            0.0,
            state.pending_workload_cpu - assigned_workload_cpu,
        )
        assigned_workload = (
            assigned_workload_cpu / state.total_cpu_capacity
        )
        unmet_workload = (
            unmet_workload_cpu / state.total_cpu_capacity
        )
        assigned_gpu_demand = state.gpu_demand
        unmet_gpu_demand = max(
            0.0, state.requested_gpu_demand - assigned_gpu_demand
        )

        # PUE is undefined when no IT load is running. Store NaN instead of
        # dividing by an epsilon, which produced extremely large fake values.
        pue = (
            state.total_power_kw / state.it_power_kw
            if state.it_power_kw > 1e-9
            else float("nan")
        )
        temperature_violation = int(
            state.temperature < state.min_temp_c
            or state.temperature > state.max_temp_c
        )
        power_cap_violation = int(
            state.total_power_kw > state.facility_power_capacity_kw
        )
        operating_limit_violation = int(
            state.total_power_kw > state.facility_operating_limit_kw
        )
        if state.power_risk_ratio >= state.power_risk_critical_fraction:
            power_risk_level = "critical"
        elif state.power_risk_ratio >= state.power_risk_warning_fraction:
            power_risk_level = "warning"
        else:
            power_risk_level = "normal"

        deadline_missed_cpu = advance_batch_queue()

        result = {
            "hour": hour,
            "timestep_h": dt,
            "workload_source": (
                "cloud_trace" if trace_arrivals is not None else "profile"
            ),
            "trace_bin_start": (
                trace_arrivals[hour]["bin_start"]
                if trace_arrivals is not None
                else None
            ),
            "hardware_profile": state.hardware_profile_name,
            "processor_model": state.processor_model,
            "accelerator_model": state.accelerator_model,
            "processing_unit_name": state.processing_unit_name,
            "servers_per_cluster": state.servers_per_cluster,
            "total_server_count": state.total_server_count,
            "processing_capacity_per_server": (
                state.processing_capacity_per_server
            ),
            "server_idle_power_kw": state.server_idle_power_kw,
            "server_max_power_kw": state.server_max_power_kw,
            "gpu_server_count": state.gpu_server_count,
            "gpus_per_gpu_server": state.gpus_per_gpu_server,
            "total_gpu_capacity": state.total_gpu_capacity,
            "gpu_idle_power_kw": state.gpu_idle_power_kw,
            "gpu_max_power_kw": state.gpu_max_power_kw,
            "workload": state.pending_workload_fraction,
            "workload_cpu_units": state.pending_workload_cpu,
            "total_cpu_capacity": state.total_cpu_capacity,
            "assigned_workload": assigned_workload,
            "assigned_workload_cpu_units": assigned_workload_cpu,
            "unmet_workload": unmet_workload,
            "unmet_workload_cpu_units": unmet_workload_cpu,
            "interactive_workload_cpu_units": state.interactive_workload_cpu,
            "batch_arrival_cpu_units": state.batch_arrival_cpu,
            "batch_served_cpu_units": state.batch_served_cpu,
            "batch_backlog_cpu_units": state.batch_backlog_cpu,
            "batch_deadline_missed_cpu_units": deadline_missed_cpu,
            "batch_arrival_gpu_hours": state.batch_arrival_gpu_work,
            "batch_backlog_gpu_hours": state.batch_backlog_gpu_work,
            "scheduled_gpu_hours": state.scheduled_gpu_work,
            "requested_average_gpus": state.requested_gpu_demand,
            "assigned_average_gpus": assigned_gpu_demand,
            "unmet_average_gpus": unmet_gpu_demand,
            "active_gpu_count": state.active_gpu_count,
            "gpu_utilisation": state.gpu_utilisation,
            "agent_response_time_s": response_time,
            "agent_timed_out": int(timed_out),
            "fallback_used": int(fallback_used),
            "scheduling_ai_skipped": int(scheduling_ai_skipped),
            "agentic_architecture": (
                agentic_architecture if is_agentic else "not_applicable"
            ),
            "agent_prompt_tokens": int(
                token_usage.get("prompt_tokens", 0) or 0
            ),
            "agent_completion_tokens": int(
                token_usage.get("completion_tokens", 0) or 0
            ),
            "agent_total_tokens": int(
                token_usage.get("total_tokens", 0) or 0
            ),
            "IT_power_kw": state.it_power_kw,
            "CPU_server_power_kw": state.cpu_it_power_kw,
            "accelerator_power_kw": state.accelerator_power_kw,
            "cooling_power_kw": state.cooling_power_kw,
            "cooling_heat_removed_kw": state.cooling_heat_removed_kw,
            "effective_cooling_cop": state.effective_cooling_cop,
            "cooling_setpoint_C": state.cooling_setpoint_c,
            "ambient_temperature_C": state.ambient_temperature,
            "total_power_kw": state.total_power_kw,
            "IT_energy_kwh": state.it_power_kw * dt,
            "cooling_energy_kwh": state.cooling_power_kw * dt,
            "total_energy_kwh": state.total_power_kw * dt,
            "solar_kw": state.solar_kw,
            "solar_used_kwh": solar_used_kw * dt,
            "solar_to_load_kwh": state.solar_to_load_kw * dt,
            "solar_to_battery_kwh": state.solar_to_battery_kw * dt,
            "solar_curtailed_kwh": state.solar_curtailed_kw * dt,
            "battery_command_kw": state.battery_command_kw,
            "battery_power_kw": state.battery_power_kw,
            "battery_discharge_kwh": discharge_power_kw * dt,
            "battery_charge_kwh": charge_power_kw * dt,
            "battery_SOC": state.battery_soc,
            "battery_capex_per_kwh": state.battery_capex_per_kwh,
            "battery_cycle_life": state.battery_cycle_life,
            "battery_degradation_cost_per_kwh": (
                state.battery_degradation_cost_per_kwh
            ),
            "battery_degradation_cost": state.battery_degradation_cost,
            "battery_equivalent_full_cycles": (
                state.battery_equivalent_full_cycles
            ),
            "grid_power_kw": state.grid_power_kw,
            "facility_power_capacity_kw": state.facility_power_capacity_kw,
            "facility_operating_limit_kw": state.facility_operating_limit_kw,
            "power_risk_ratio": state.power_risk_ratio,
            "power_risk_level": power_risk_level,
            "power_cap_violation": power_cap_violation,
            "operating_limit_violation": operating_limit_violation,
            "grid_energy_kwh": state.grid_power_kw * dt,
            "grid_to_load_kwh": state.grid_to_load_kw * dt,
            "grid_to_battery_kwh": state.grid_to_battery_kw * dt,
            "temperature_C": state.temperature,
            "temperature_violation": temperature_violation,
            "electricity_price": state.grid_price,
            "grid_energy_cost": state.grid_energy_cost,
            "counterfactual_grid_cost_without_battery": (
                state.counterfactual_grid_cost_without_battery
            ),
            "battery_net_saving_vs_grid": (
                state.battery_net_saving_vs_grid
            ),
            "cost": state.cost,
            "pue": pue,
        }
        results.append(result)

        print(
            f"Hour {hour:03d} | "
            f"Load={result['workload']:.2f} | "
            f"IT={result['IT_power_kw']:.2f} kW | "
            f"Grid={result['grid_power_kw']:.2f} kW | "
            f"Solar={result['solar_kw']:.2f} kW | "
            f"SOC={result['battery_SOC'] * 100:.1f}% | "
            f"Temp={result['temperature_C']:.2f} C | "
            f"Cost=${result['cost']:.2f}"
        )

    return results
