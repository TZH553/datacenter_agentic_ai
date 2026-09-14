from state import state, reset_state

from workload import add_workload

import time

from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError
)

import pandas as pd

from state import state, reset_state

from models import (
    calculate_it_power,
    calculate_cooling_power,
    update_temperature,
    update_battery,
    calculate_grid_power,
    calculate_cost
)

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

from concurrent.futures import ThreadPoolExecutor, TimeoutError

from controllers import RuleBasedController

def run_crew_with_timeout(
    data_center_crew,
    timeout_seconds=60
):

    executor = ThreadPoolExecutor(
        max_workers=1
    )

    future = executor.submit(
        data_center_crew.kickoff
    )

    try:

        result = future.result(
            timeout=timeout_seconds
        )

        executor.shutdown(
            wait=False
        )

        return result, False

    except TimeoutError:

        print(
            f"[WARNING] CrewAI exceeded "
            f"{timeout_seconds} seconds."
        )

        future.cancel()

        executor.shutdown(
            wait=False,
            cancel_futures=True
        )

        return None, True
    
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

            from crew import data_center_crew

            print(
                f"\n[AGENTIC] Running CrewAI "
                f"for hour {hour}..."
            )

            data_center_crew.kickoff()

            # Import here so baseline simulations don't
            # initialise CrewAI unnecessarily.
            start_time = time.perf_counter()


            crew_result, timed_out = (
                run_crew_with_timeout(
                    data_center_crew,
                    timeout_seconds=60
                )
            )

            response_time = time.perf_counter() - start_time

            print(
                f"CrewAI execution time: "
                f"{response_time:.2f} seconds"
            )

            if timed_out:

                print(
                    "[FALLBACK] Using rule-based action."
                )

                fallback_controller = (
                    RuleBasedController()
                )

                action = (
                    fallback_controller.decide()
                )

            else:
                print(
                f"[AGENTIC] CrewAI completed "
                f"in {response_time:.2f} seconds."
                )


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