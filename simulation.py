from state import state, reset_state

from workload import add_workload

import time

from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError
)

from state import state, reset_state

from controllers import RuleBasedController

from control import apply_action

from models import (
    calculate_it_power,
    calculate_cooling_power,
    update_temperature,
    update_battery,
    calculate_grid_power,
    calculate_cost
)

from controllers import RuleBasedController

import copy
import multiprocessing as mp
import time
import traceback

def _crew_worker(initial_state, result_queue):
    """
    Execute CrewAI inside an isolated child process.

    CrewAI tools modify the child's copy of `state`. If execution
    succeeds, the updated state is returned to the parent process.
    """
    try:
        # Restore the simulation state inside the child.
        state.__dict__.clear()
        state.__dict__.update(copy.deepcopy(initial_state))

        # Import here so each spawned process initializes CrewAI locally.
        from crew import data_center_crew

        crew_result = data_center_crew.kickoff()

        result_queue.put({
            "status": "success",
            "state": copy.deepcopy(state.__dict__),
            "crew_result": str(crew_result),
        })

    except Exception:
        result_queue.put({
            "status": "error",
            "error": traceback.format_exc(),
        })


def run_crew_with_timeout(timeout_seconds=60):
    """
    Run CrewAI in a process that can be terminated safely.

    Returns:
        crew_result, timed_out, error_message
    """
    context = mp.get_context("spawn")
    result_queue = context.Queue()

    initial_state = copy.deepcopy(state.__dict__)

    process = context.Process(
        target=_crew_worker,
        args=(initial_state, result_queue),
    )

    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        print(
            f"[WARNING] CrewAI exceeded "
            f"{timeout_seconds} seconds."
        )

        process.terminate()
        process.join(timeout=5)

        # Escalate if terminate() did not stop it.
        if process.is_alive():
            process.kill()
            process.join()

        result_queue.close()

        return None, True, None

    if result_queue.empty():
        result_queue.close()

        return (
            None,
            False,
            "CrewAI process exited without returning a result.",
        )

    message = result_queue.get()
    result_queue.close()

    if message["status"] == "error":
        return None, False, message["error"]

    # Copy the successful child's state into the parent simulation.
    state.__dict__.clear()
    state.__dict__.update(message["state"])

    return message["crew_result"], False, None

def run_simulation(
    controller,
    workload_profile,
    solar_profile,
    price_profile,
    hours,
    is_agentic=False
):

    # ==========================================
    # RESET BEFORE EVERY EXPERIMENT
    # ==========================================

    reset_state()

    results = []

    print()
    print("=" * 60)

    if is_agentic:
        print("Starting CrewAI Agentic EMS")
    else:
        print(
            f"Starting {controller.name}"
        )

    print("=" * 60)


    for hour in range(hours):

        # ======================================
        # ENVIRONMENT
        # ======================================

        state.solar_kw = float(
            solar_profile[hour]
        )

        state.grid_price = float(
            price_profile[hour]
        )


        # ======================================
        # WORKLOAD
        # ======================================

        add_workload(
            workload_profile[hour]
        )


        # ======================================
        # RESET HOURLY COMMANDS
        # ======================================

        state.battery_command_kw = 0.0


        # ======================================
        # CONTROLLER
        # ======================================

        response_time = 0.0
        timed_out = False
        fallback_used = False

        if is_agentic:
            print(
                f"\n[AGENTIC] Running CrewAI "
                f"for hour {hour}..."
            )

            start_time = time.perf_counter()

            crew_result, timed_out, crew_error = (
                run_crew_with_timeout(
                    timeout_seconds=60
                )
            )

            response_time = (
                time.perf_counter() - start_time
            )

            if timed_out:
                print(
                    "[FALLBACK] CrewAI timed out. "
                    "Using rule-based controller."
                )

                fallback_used = True

                action = RuleBasedController().decide()
                apply_action(action)

            elif crew_error is not None:
                print(
                    "[FALLBACK] CrewAI execution failed:"
                )
                print(crew_error)

                fallback_used = True

                action = RuleBasedController().decide()
                apply_action(action)

            else:
                print(
                    f"[AGENTIC] CrewAI completed in "
                    f"{response_time:.2f} seconds."
                )

        else:
            action = controller.decide()
            apply_action(action)


        # ======================================
        # PHYSICAL SIMULATION
        # ======================================

        calculate_it_power()

        calculate_cooling_power()

        state.total_power_kw = (
            state.it_power_kw
            + state.cooling_power_kw
        )

        update_temperature()

        update_battery(
            dt=1.0
        )

        calculate_grid_power()

        calculate_cost(
            dt=1.0
        )


        # ======================================
        # ADDITIONAL ENERGY METRICS
        # ======================================

        # Battery discharge:
        # positive battery power means battery supplying load
        battery_discharge_kwh = max(
            0.0,
            state.battery_power_kw
        )

        # Battery charge:
        # negative battery power means battery charging
        battery_charge_kwh = max(
            0.0,
            -state.battery_power_kw
        )


        # Solar actually used by the data centre
        # or battery charging.
        solar_used_kw = min(
            state.solar_kw,
            state.total_power_kw
            + battery_charge_kwh
        )

        # Solar that could not be used
        solar_curtailed_kw = max(
            0.0,
            state.solar_kw
            - solar_used_kw
        )


        # ======================================
        # OTHER METRICS
        # ======================================

        pue = (
            state.total_power_kw
            / max(
                state.it_power_kw,
                0.0001
            )
        )

        temperature_violation = int(
            state.temperature < 19
            or state.temperature > 27
        )


        # ======================================
        # STORE RESULT FOR THIS HOUR
        # ======================================

        result = {

            "hour":
                hour,

            "workload":
                float(workload_profile[hour]),


            # ==========================================
            # AGENT METRICS
            # ==========================================

            "agent_response_time_s":
                response_time,

            "agent_timed_out":
                int(timed_out),

            "fallback_used":
                int(fallback_used),


            # ==========================================
            # POWER
            # ==========================================

            "IT_power_kw":
                state.it_power_kw,

            "cooling_power_kw":
                state.cooling_power_kw,

            "total_power_kw":
                state.total_power_kw,

            "total_energy_kwh":
                state.total_power_kw,


            # ==========================================
            # SOLAR
            # ==========================================

            "solar_kw":
                state.solar_kw,

            "solar_used_kwh":
                solar_used_kw,

            "solar_curtailed_kwh":
                solar_curtailed_kw,


            # ==========================================
            # BATTERY
            # ==========================================

            "battery_command_kw":
                state.battery_command_kw,

            "battery_power_kw":
                state.battery_power_kw,

            "battery_discharge_kwh":
                battery_discharge_kwh,

            "battery_charge_kwh":
                battery_charge_kwh,

            "battery_SOC":
                state.battery_soc,


            # ==========================================
            # GRID
            # ==========================================

            "grid_power_kw":
                state.grid_power_kw,

            "grid_energy_kwh":
                state.grid_power_kw,


            # ==========================================
            # THERMAL
            # ==========================================

            "temperature_C":
                state.temperature,

            "temperature_violation":
                temperature_violation,


            # ==========================================
            # COST
            # ==========================================

            "electricity_price":
                state.grid_price,

            "cost":
                state.cost,


            # ==========================================
            # EFFICIENCY
            # ==========================================

            "pue":
                pue
        }

        results.append(result)


        # ======================================
        # HOURLY OUTPUT
        # ======================================

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