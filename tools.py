from typing import Dict, List, Tuple

from crewai.tools import tool

from models import calculate_host_power
from state import state


def _normalise_allocations(
    allocations: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    """Validate a batch allocation and fill omitted hosts with zero."""
    unknown = set(allocations) - set(state.hosts)
    if unknown:
        raise ValueError(f"Unknown hosts: {sorted(unknown)}")

    normalised = {}
    assigned_cpu = 0.0
    for name, host in state.hosts.items():
        utilisation = float(allocations.get(name, 0.0))
        if not 0.0 <= utilisation <= 1.0:
            raise ValueError(
                f"{name} utilisation must be between 0 and 1."
            )
        normalised[name] = utilisation
        assigned_cpu += utilisation * host["cpu_capacity"]

    tolerance_cpu = max(1.0, state.pending_workload_cpu * 0.005)
    difference = assigned_cpu - state.pending_workload_cpu
    if abs(difference) > tolerance_cpu:
        raise ValueError(
            f"Assigned {assigned_cpu:.2f} CPU units but demand is "
            f"{state.pending_workload_cpu:.2f}; difference "
            f"{difference:+.2f} exceeds tolerance {tolerance_cpu:.2f}."
        )

    return normalised, assigned_cpu


def _set_allocations(
    allocations: Dict[str, float],
    derive_power_states: bool,
) -> float:
    normalised, assigned_cpu = _normalise_allocations(allocations)
    for name, utilisation in normalised.items():
        state.hosts[name]["utilisation"] = utilisation
        if derive_power_states:
            state.hosts[name]["active"] = utilisation > 0.0
    return assigned_cpu


def _set_active_hosts(active_hosts: List[str]) -> None:
    unknown = set(active_hosts) - set(state.hosts)
    if unknown:
        raise ValueError(f"Unknown hosts: {sorted(unknown)}")

    active_set = set(active_hosts)
    loaded_but_inactive = [
        name
        for name, host in state.hosts.items()
        if host["utilisation"] > 0.0 and name not in active_set
    ]
    if loaded_but_inactive:
        raise ValueError(
            "Loaded hosts must remain active: "
            f"{loaded_but_inactive}"
        )

    for name in state.hosts:
        state.hosts[name]["active"] = name in active_set


def _normalise_cooling(cooling_factor: float) -> float:
    value = float(cooling_factor)
    if not 0.8 <= value <= 1.5:
        raise ValueError("Cooling factor must be between 0.8 and 1.5.")
    return value


def _normalise_battery(power_kw: float) -> float:
    value = float(power_kw)
    if not (
        -state.battery_max_charge_kw
        <= value
        <= state.battery_max_discharge_kw
    ):
        raise ValueError(
            "Battery command exceeds charge/discharge power limits."
        )
    if value > 0 and state.battery_soc <= state.battery_min_soc:
        raise ValueError("Battery SOC is too low for discharge.")
    if value < 0 and state.battery_soc >= state.battery_max_soc:
        raise ValueError("Battery SOC is too high for charging.")
    return value


def _set_cooling(cooling_factor: float) -> None:
    state.cooling_factor = _normalise_cooling(cooling_factor)


def _set_battery(power_kw: float) -> None:
    state.battery_command_kw = _normalise_battery(power_kw)


@tool("Get cluster telemetry")
def get_cluster_telemetry() -> str:
    """Return all compute, thermal, renewable, grid, and battery telemetry."""
    host_data = []
    for name, host in state.hosts.items():
        assigned_cpu = host["utilisation"] * host["cpu_capacity"]
        host_data.append(
            {
                "host": name,
                "active": host["active"],
                "cpu_capacity": host["cpu_capacity"],
                "assigned_cpu_units": round(assigned_cpu, 2),
                "utilisation": round(host["utilisation"], 4),
                "power_kw": round(calculate_host_power(host), 2),
            }
        )

    return str(
        {
            "hosts": host_data,
            "total_cpu_capacity": state.total_cpu_capacity,
            "pending_workload_cpu_units": round(
                state.pending_workload_cpu, 2
            ),
            "temperature_C": round(state.temperature, 2),
            "minimum_temperature_C": state.min_temp_c,
            "maximum_temperature_C": state.max_temp_c,
            "IT_power_kw": round(state.it_power_kw, 2),
            "cooling_power_kw": round(state.cooling_power_kw, 2),
            "cooling_factor": state.cooling_factor,
            "effective_cooling_cop": round(
                state.effective_cooling_cop, 2
            ),
            "solar_kw": round(state.solar_kw, 2),
            "grid_price": state.grid_price,
            "battery_SOC_percent": round(
                state.battery_soc * 100, 2
            ),
            "battery_command_kw": state.battery_command_kw,
        }
    )


@tool("Schedule all workload")
def schedule_workload_batch(
    allocations: Dict[str, float],
) -> str:
    """Set every host utilisation in one call.

    allocations maps host names to utilisation fractions from 0 to 1.
    Omitted hosts are assigned zero. Total assigned CPU must equal demand.
    """
    try:
        assigned_cpu = _set_allocations(
            allocations, derive_power_states=False
        )
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"

    return (
        f"ACCEPTED: assigned {assigned_cpu:.2f} CPU units in one batch. "
        "Finish this task; do not repeat the call."
    )


@tool("Set all host power states")
def set_host_power_batch(active_hosts: List[str]) -> str:
    """Set all host power states in one call.

    active_hosts lists hosts that must remain on. All omitted hosts turn off.
    Loaded hosts cannot be omitted.
    """
    try:
        _set_active_hosts(active_hosts)
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"

    return (
        f"ACCEPTED: active hosts are {active_hosts}. "
        "Finish this task; do not repeat the call."
    )


@tool("Apply complete compute plan")
def apply_compute_plan(
    allocations: Dict[str, float],
) -> str:
    """Atomically set all utilisation and matching power states.

    Omitted hosts receive zero utilisation and turn off. Hosts with positive
    utilisation turn on. Total assigned CPU must equal pending demand.
    """
    try:
        assigned_cpu = _set_allocations(
            allocations, derive_power_states=True
        )
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"

    active_hosts = [
        name for name, host in state.hosts.items() if host["active"]
    ]
    return (
        f"ACCEPTED: assigned {assigned_cpu:.2f} CPU units; active hosts "
        f"are {active_hosts}. Finish this task; do not repeat the call."
    )


@tool("Set cooling level")
def set_cooling_level(cooling_factor: float) -> str:
    """Set relative cooling capacity from 0.8 to 1.5."""
    try:
        _set_cooling(cooling_factor)
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"
    return (
        f"ACCEPTED: cooling factor is {state.cooling_factor:.2f}. "
        "Finish this task; do not repeat the call."
    )


@tool("Dispatch battery")
def dispatch_battery(power_kw: float) -> str:
    """Set battery power; positive discharges and negative charges."""
    try:
        _set_battery(power_kw)
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"
    return (
        f"ACCEPTED: battery command is "
        f"{state.battery_command_kw:.2f} kW. "
        "Finish this task; do not repeat the call."
    )


@tool("Apply complete EMS plan")
def apply_ems_plan(
    allocations: Dict[str, float],
    cooling_factor: float,
    battery_power_kw: float,
) -> str:
    """Atomically apply compute, cooling, and battery decisions in one call."""
    try:
        normalised, assigned_cpu = _normalise_allocations(allocations)
        normalised_cooling = _normalise_cooling(cooling_factor)
        normalised_battery = _normalise_battery(battery_power_kw)
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"

    # Mutate state only after every part of the plan has validated.
    for name, utilisation in normalised.items():
        state.hosts[name]["utilisation"] = utilisation
        state.hosts[name]["active"] = utilisation > 0.0
    state.cooling_factor = normalised_cooling
    state.battery_command_kw = normalised_battery

    active_hosts = [
        name for name, host in state.hosts.items() if host["active"]
    ]
    return (
        f"ACCEPTED: {assigned_cpu:.2f} CPU units assigned; active hosts "
        f"{active_hosts}; cooling factor {state.cooling_factor:.2f}; "
        f"battery {state.battery_command_kw:.2f} kW. "
        "The complete plan is applied. Finish now."
    )


# Retained for manual testing, but no AI agent uses these fine-grained tools.
@tool("Schedule one host")
def schedule_task(host_name: str, utilisation: float) -> str:
    """Set one host utilisation for manual diagnostics."""
    if host_name not in state.hosts:
        return f"ERROR: {host_name} does not exist."
    utilisation = float(utilisation)
    if not 0.0 <= utilisation <= 1.0:
        return "ERROR: utilisation must be between 0 and 1."
    state.hosts[host_name]["utilisation"] = utilisation
    return f"{host_name} utilisation set to {utilisation:.4f}."


@tool("Set one host power")
def set_host_power(host_name: str, active: bool) -> str:
    """Set one host power state for manual diagnostics."""
    if host_name not in state.hosts:
        return f"ERROR: {host_name} does not exist."
    host = state.hosts[host_name]
    if not active and host["utilisation"] > 0.0:
        return f"REJECTED: {host_name} still has workload."
    host["active"] = bool(active)
    return f"{host_name} power state set to {host['active']}."
