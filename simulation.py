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
from workload import add_workload


def _crew_worker(initial_state, result_queue):
    """Execute CrewAI in an isolated process and return its updated state."""
    try:
        state.__dict__.clear()
        state.__dict__.update(copy.deepcopy(initial_state))

        from crew import data_center_crew

        crew_result = data_center_crew.kickoff()
        result_queue.put(
            {
                "status": "success",
                "state": copy.deepcopy(state.__dict__),
                "crew_result": str(crew_result),
            }
        )
    except Exception:
        result_queue.put(
            {
                "status": "error",
                "error": traceback.format_exc(),
            }
        )


def run_crew_with_timeout(timeout_seconds=60):
    """Run CrewAI in a child process that can be stopped on timeout."""
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_crew_worker,
        args=(copy.deepcopy(state.__dict__), result_queue),
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
        return None, True, None

    try:
        message = result_queue.get(timeout=2)
    except queue.Empty:
        result_queue.close()
        return None, False, "CrewAI exited without returning a result."

    result_queue.close()

    if message["status"] == "error":
        return None, False, message["error"]

    state.__dict__.clear()
    state.__dict__.update(message["state"])
    return message["crew_result"], False, None


def run_simulation(
    controller,
    workload_profile,
    solar_profile,
    price_profile,
    hours,
    is_agentic=False,
    config=None,
):
    """Run all power and energy calculations using one consistent timestep."""
    reset_state(config)
    results = []
    dt = state.timestep_h

    print()
    print("=" * 60)
    print("Starting CrewAI Agentic EMS" if is_agentic else f"Starting {controller.name}")
    print("=" * 60)

    if any(len(profile) < hours for profile in (
        workload_profile, solar_profile, price_profile
    )):
        raise ValueError("Every input profile must contain at least 'hours' values.")

    for hour in range(hours):
        state.solar_kw = float(solar_profile[hour])
        state.grid_price = float(price_profile[hour])
        add_workload(workload_profile[hour])
        state.battery_command_kw = 0.0

        response_time = 0.0
        timed_out = False
        fallback_used = False

        if is_agentic:
            print(f"\n[AGENTIC] Running CrewAI for hour {hour}...")
            start_time = time.perf_counter()
            _, timed_out, crew_error = run_crew_with_timeout(
                timeout_seconds=60
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

        pue = state.total_power_kw / max(state.it_power_kw, 1e-9)
        temperature_violation = int(
            state.temperature < state.min_temp_c
            or state.temperature > state.max_temp_c
        )

        result = {
            "hour": hour,
            "timestep_h": dt,
            "workload": state.pending_workload_fraction,
            "workload_cpu_units": state.pending_workload_cpu,
            "total_cpu_capacity": state.total_cpu_capacity,
            "assigned_workload": assigned_workload,
            "assigned_workload_cpu_units": assigned_workload_cpu,
            "unmet_workload": unmet_workload,
            "unmet_workload_cpu_units": unmet_workload_cpu,
            "agent_response_time_s": response_time,
            "agent_timed_out": int(timed_out),
            "fallback_used": int(fallback_used),
            "IT_power_kw": state.it_power_kw,
            "cooling_power_kw": state.cooling_power_kw,
            "cooling_heat_removed_kw": state.cooling_heat_removed_kw,
            "effective_cooling_cop": state.effective_cooling_cop,
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
            "grid_power_kw": state.grid_power_kw,
            "grid_energy_kwh": state.grid_power_kw * dt,
            "grid_to_load_kwh": state.grid_to_load_kw * dt,
            "grid_to_battery_kwh": state.grid_to_battery_kw * dt,
            "temperature_C": state.temperature,
            "temperature_violation": temperature_violation,
            "electricity_price": state.grid_price,
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
