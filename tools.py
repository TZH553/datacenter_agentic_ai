from crewai.tools import tool

from models import calculate_host_power
from state import state


@tool("Get cluster telemetry")
def get_cluster_telemetry() -> str:
    """Return workload, heterogeneous host capacity, and energy telemetry."""
    host_data = []
    for name, host in state.hosts.items():
        assigned_cpu = host["utilisation"] * host["cpu_capacity"]
        host_data.append(
            {
                "host": name,
                "active": host["active"],
                "cpu_capacity": host["cpu_capacity"],
                "assigned_cpu_units": round(assigned_cpu, 2),
                "utilisation_percent": round(host["utilisation"] * 100, 2),
                "power_kw": round(calculate_host_power(host), 2),
            }
        )

    telemetry = {
        "hosts": host_data,
        "total_cpu_capacity": state.total_cpu_capacity,
        "pending_workload_fraction": state.pending_workload_fraction,
        "pending_workload_cpu_units": state.pending_workload_cpu,
        "IT_power_kw": round(state.it_power_kw, 2),
        "cooling_power_kw": round(state.cooling_power_kw, 2),
        "cooling_heat_removed_kw": round(
            state.cooling_heat_removed_kw, 2
        ),
        "effective_cooling_cop": round(state.effective_cooling_cop, 2),
        "temperature_C": round(state.temperature, 2),
        "solar_kw": round(state.solar_kw, 2),
        "grid_price": state.grid_price,
        "battery_SOC_percent": round(state.battery_soc * 100, 2),
        "grid_power_kw": round(state.grid_power_kw, 2),
        "solar_to_load_kw": round(state.solar_to_load_kw, 2),
        "solar_to_battery_kw": round(state.solar_to_battery_kw, 2),
        "grid_to_load_kw": round(state.grid_to_load_kw, 2),
        "grid_to_battery_kw": round(state.grid_to_battery_kw, 2),
    }
    return str(telemetry)


@tool("Schedule workload")
def schedule_task(host_name: str, utilisation: float) -> str:
    """Assign a utilization fraction to one host."""
    if host_name not in state.hosts:
        return f"ERROR: {host_name} does not exist."

    utilisation = float(utilisation)
    if not 0.0 <= utilisation <= 1.0:
        return "ERROR: utilisation must be between 0 and 1."

    host = state.hosts[host_name]
    if not host["active"]:
        return f"ERROR: {host_name} is powered OFF."

    host["utilisation"] = utilisation
    assigned_cpu = utilisation * host["cpu_capacity"]
    return (
        f"{host_name} assigned {assigned_cpu:.2f} CPU units "
        f"({utilisation * 100:.1f}% utilisation)."
    )


@tool("Set host power")
def set_host_power(host_name: str, active: bool) -> str:
    """Power a host on or off; a loaded host cannot be powered off."""
    if host_name not in state.hosts:
        return f"ERROR: {host_name} does not exist."

    host = state.hosts[host_name]
    if not active and host["utilisation"] > 0.01:
        return (
            f"REJECTED: {host_name} still has "
            f"{host['utilisation'] * host['cpu_capacity']:.2f} CPU units. "
            "Migrate its workload first."
        )

    host["active"] = active
    return f"{host_name} power state = {'ON' if active else 'OFF'}."


@tool("Set cooling level")
def set_cooling_level(cooling_factor: float) -> str:
    """Set relative cooling effort from 0.8 to 1.5."""
    cooling_factor = float(cooling_factor)
    if not 0.8 <= cooling_factor <= 1.5:
        return "REJECTED: cooling factor must be between 0.8 and 1.5."

    state.cooling_factor = cooling_factor
    return f"Cooling factor changed to {cooling_factor:.2f}."


@tool("Dispatch battery")
def dispatch_battery(power_kw: float) -> str:
    """Set battery power: positive discharges, negative charges."""
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
    if power_kw > 0 and state.battery_soc <= state.battery_min_soc:
        return "REJECTED: battery SOC too low for discharge."
    if power_kw < 0 and state.battery_soc >= state.battery_max_soc:
        return "REJECTED: battery SOC too high for charging."

    state.battery_command_kw = power_kw
    mode = "DISCHARGE" if power_kw > 0 else "CHARGE" if power_kw < 0 else "IDLE"
    return (
        f"Battery command accepted. Mode={mode}, "
        f"power={power_kw:.2f} kW."
    )
