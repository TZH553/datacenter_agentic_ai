from crewai.tools import tool


from state import state
from models import calculate_host_power


# =========================================================
# TELEMETRY TOOL
# =========================================================

@tool("Get cluster telemetry")
def get_cluster_telemetry() -> str:
    """
    Return the current data-centre operating state including
    server utilisation, IT power, temperature, solar generation,
    electricity price and battery SOC.
    """

    host_data = []

    for name, host in state.hosts.items():

        host_data.append({

            "host": name,

            "active":
                host["active"],

            "utilisation_percent":
                round(
                    host["utilisation"] * 100,
                    2
                ),

            "power_kw":
                round(
                    calculate_host_power(host),
                    2
                )
        })


    telemetry = {

        "hosts":
            host_data,

        "pending_workload":
            state.pending_workload,

        "IT_power_kw":
            round(
                state.it_power_kw,
                2
            ),

        "cooling_power_kw":
            round(
                state.cooling_power_kw,
                2
            ),

        "temperature_C":
            round(
                state.temperature,
                2
            ),

        "solar_kw":
            round(
                state.solar_kw,
                2
            ),

        "grid_price":
            state.grid_price,

        "battery_SOC_percent":
            round(
                state.battery_soc * 100,
                2
            ),

        "grid_power_kw":
            round(
                state.grid_power_kw,
                2
            )
    }

    return str(telemetry)


# =========================================================
# WORKLOAD SCHEDULER TOOL
# =========================================================

@tool("Schedule workload")
def schedule_task(
    host_name: str,
    utilisation: float
) -> str:
    """
    Assign utilisation to a server host.

    utilisation must be between 0 and 1.
    """

    if host_name not in state.hosts:
        return (
            f"ERROR: {host_name} does not exist."
        )

    utilisation = float(utilisation)

    if utilisation < 0 or utilisation > 1:
        return (
            "ERROR: utilisation must be between "
            "0 and 1."
        )

    host = state.hosts[host_name]

    if not host["active"]:

        return (
            f"ERROR: {host_name} is powered OFF."
        )


    host["utilisation"] = utilisation


    return (
        f"{host_name} utilisation set to "
        f"{utilisation * 100:.1f}%."
    )


# =========================================================
# HOST POWER TOOL
# =========================================================

@tool("Set host power")
def set_host_power(
    host_name: str,
    active: bool
) -> str:
    """
    Power a host ON or OFF.

    Hosts containing workload cannot be powered off.
    """

    if host_name not in state.hosts:

        return (
            f"ERROR: {host_name} does not exist."
        )


    host = state.hosts[host_name]


    if not active:

        if host["utilisation"] > 0.01:

            return (
                f"REJECTED: {host_name} still has "
                f"{host['utilisation'] * 100:.1f}% "
                f"utilisation. Migrate its workload first."
            )


    host["active"] = active


    return (
        f"{host_name} power state = "
        f"{'ON' if active else 'OFF'}."
    )


# =========================================================
# COOLING TOOL
# =========================================================

@tool("Set cooling level")
def set_cooling_level(
    cooling_factor: float
) -> str:
    """
    Change cooling effort.

    Allowed range:
        0.8 to 1.5

    1.0 = normal cooling.
    """

    cooling_factor = float(
        cooling_factor
    )


    if not 0.8 <= cooling_factor <= 1.5:

        return (
            "REJECTED: cooling factor must "
            "be between 0.8 and 1.5."
        )


    state.cooling_factor = cooling_factor


    return (
        f"Cooling factor changed to "
        f"{cooling_factor:.2f}."
    )


# =========================================================
# BATTERY TOOL
# =========================================================

@tool("Dispatch battery")
def dispatch_battery(
    power_kw: float
) -> str:
    """
    Set battery dispatch command.

    Positive power:
        discharge battery

    Negative power:
        charge battery

    Zero:
        idle
    """

    power_kw = float(power_kw)


    if power_kw > state.battery_max_discharge_kw:

        return (
            f"REJECTED: maximum discharge is "
            f"{state.battery_max_discharge_kw} kW."
        )


    if power_kw < -state.battery_max_charge_kw:

        return (
            f"REJECTED: maximum charge is "
            f"{state.battery_max_charge_kw} kW."
        )


    if power_kw > 0 and state.battery_soc <= 0.10:

        return (
            "REJECTED: battery SOC too low "
            "for discharge."
        )


    if power_kw < 0 and state.battery_soc >= 0.95:

        return (
            "REJECTED: battery SOC too high "
            "for charging."
        )


    state.battery_command_kw = power_kw


    if power_kw > 0:

        mode = "DISCHARGE"

    elif power_kw < 0:

        mode = "CHARGE"

    else:

        mode = "IDLE"


    return (
        f"Battery command accepted. "
        f"Mode={mode}, "
        f"power={power_kw:.2f} kW."
    )