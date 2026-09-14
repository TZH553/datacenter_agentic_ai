from typing import Dict

from crewai.tools import tool

from models import calculate_host_power, estimate_cooling
from state import state


def _build_allocations(target_utilisation: float) -> Dict[str, float]:
    """Deterministically allocate CPU demand across heterogeneous hosts."""
    target = float(target_utilisation)
    if not 0.50 <= target <= 1.0:
        raise ValueError(
            "Target utilisation must be between 0.50 and 1.00."
        )

    ordered_hosts = sorted(
        state.hosts,
        key=lambda name: state.hosts[name]["cpu_capacity"],
        reverse=True,
    )
    allocations = {name: 0.0 for name in state.hosts}
    remaining_cpu = state.pending_workload_cpu

    # Consolidate onto larger hosts within the selected target.
    for name in ordered_hosts:
        if remaining_cpu <= 1e-9:
            break
        capacity = state.hosts[name]["cpu_capacity"]
        assigned_cpu = min(remaining_cpu, capacity * target)
        allocations[name] = assigned_cpu / capacity
        remaining_cpu -= assigned_cpu

    # Use headroom up to 100% if target capacity is insufficient.
    if remaining_cpu > 1e-9:
        for name in ordered_hosts:
            capacity = state.hosts[name]["cpu_capacity"]
            headroom_cpu = capacity * (1.0 - allocations[name])
            extra_cpu = min(remaining_cpu, headroom_cpu)
            allocations[name] += extra_cpu / capacity
            remaining_cpu -= extra_cpu
            if remaining_cpu <= 1e-9:
                break

    return allocations


def _apply_allocations(
    target_utilisation: float,
    derive_power_states: bool,
) -> float:
    allocations = _build_allocations(target_utilisation)
    assigned_cpu = 0.0
    for name, utilisation in allocations.items():
        state.hosts[name]["utilisation"] = utilisation
        if derive_power_states:
            state.hosts[name]["active"] = utilisation > 0.0
        assigned_cpu += utilisation * state.hosts[name]["cpu_capacity"]
    return assigned_cpu


def _power_off_idle_hosts() -> list[str]:
    active_hosts = []
    for name, host in state.hosts.items():
        host["active"] = host["utilisation"] > 0.0
        if host["active"]:
            active_hosts.append(name)
    return active_hosts


def _normalise_cooling(cooling_factor: float) -> float:
    value = float(cooling_factor)
    if not 0.8 <= value <= 1.5:
        raise ValueError("Cooling factor must be between 0.8 and 1.5.")
    return value


def _projected_facility_power(
    allocations: Dict[str, float],
    cooling_factor: float,
) -> float:
    it_power_kw = sum(
        host["p_idle"]
        + allocations[name] * (host["p_max"] - host["p_idle"])
        for name, host in state.hosts.items()
        if allocations[name] > 0.0
    )
    cooling_power_kw, _, _ = estimate_cooling(
        it_power_kw,
        cooling_factor,
    )
    return it_power_kw + cooling_power_kw


def _normalise_battery(
    power_kw: float,
    allocations: Dict[str, float] | None = None,
    cooling_factor: float | None = None,
) -> float:
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

    if value < 0:
        if allocations is None:
            allocations = {
                name: host["utilisation"]
                for name, host in state.hosts.items()
            }
        if cooling_factor is None:
            cooling_factor = state.cooling_factor

        facility_power_kw = _projected_facility_power(
            allocations,
            cooling_factor,
        )
        solar_surplus_kw = max(
            0.0,
            state.solar_kw - facility_power_kw,
        )
        requested_charge_kw = abs(value)
        grid_charge_kw = max(
            0.0,
            requested_charge_kw - solar_surplus_kw,
        )

        if grid_charge_kw > 1e-9 and state.grid_price >= 0.18:
            raise ValueError(
                f"Charging would use {grid_charge_kw:.2f} kW from the "
                f"grid at ${state.grid_price:.2f}/kWh. Grid charging is "
                "allowed only below $0.18/kWh."
            )

    return value


@tool("Get cluster telemetry")
def get_cluster_telemetry() -> str:
    """Return all compute, thermal, renewable, grid, and battery telemetry."""
    host_data = []
    for name, host in state.hosts.items():
        host_data.append(
            {
                "host": name,
                "active": host["active"],
                "cpu_capacity": host["cpu_capacity"],
                "assigned_cpu_units": round(
                    host["utilisation"] * host["cpu_capacity"], 2
                ),
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
            "temperature_range_C": [
                state.min_temp_c,
                state.max_temp_c,
            ],
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
def schedule_workload_batch(target_utilisation: float) -> str:
    """Allocate all CPU demand deterministically in one call.

    The agent chooses only a preferred target utilisation from 0.50 to 1.00.
    Deterministic code calculates every per-host allocation.
    """
    try:
        assigned_cpu = _apply_allocations(
            target_utilisation,
            derive_power_states=False,
        )
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"

    return (
        f"ACCEPTED: assigned {assigned_cpu:.2f} CPU units with target "
        f"{float(target_utilisation):.2f}. Finish this task now."
    )


@tool("Set all host power states")
def set_host_power_batch() -> str:
    """Turn on loaded hosts and turn off all zero-utilisation hosts."""
    active_hosts = _power_off_idle_hosts()
    return (
        f"ACCEPTED: active hosts are {active_hosts}. "
        "Finish this task now."
    )


@tool("Apply complete compute plan")
def apply_compute_plan(target_utilisation: float) -> str:
    """Allocate all CPU demand and derive every host power state."""
    try:
        assigned_cpu = _apply_allocations(
            target_utilisation,
            derive_power_states=True,
        )
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"

    active_hosts = [
        name for name, host in state.hosts.items() if host["active"]
    ]
    return (
        f"ACCEPTED: assigned {assigned_cpu:.2f} CPU units; active hosts "
        f"are {active_hosts}. Finish this task now."
    )


@tool("Set cooling level")
def set_cooling_level(cooling_factor: float) -> str:
    """Set relative cooling capacity from 0.8 to 1.5."""
    try:
        state.cooling_factor = _normalise_cooling(cooling_factor)
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"
    return (
        f"ACCEPTED: cooling factor is {state.cooling_factor:.2f}. "
        "Finish this task now."
    )


@tool("Dispatch battery")
def dispatch_battery(power_kw: float) -> str:
    """Set battery power; positive discharges and negative charges."""
    try:
        state.battery_command_kw = _normalise_battery(power_kw)
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"
    return (
        f"ACCEPTED: battery command is "
        f"{state.battery_command_kw:.2f} kW. Finish this task now."
    )


@tool("Apply complete EMS plan")
def apply_ems_plan(
    target_utilisation: float,
    cooling_factor: float,
    battery_power_kw: float,
) -> str:
    """Apply compute, cooling, and battery decisions atomically.

    All inputs are simple scalars for reliable local-model tool calling.
    """
    try:
        allocations = _build_allocations(target_utilisation)
        normalised_cooling = _normalise_cooling(cooling_factor)
        normalised_battery = _normalise_battery(
            battery_power_kw,
            allocations=allocations,
            cooling_factor=normalised_cooling,
        )
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"

    assigned_cpu = 0.0
    for name, utilisation in allocations.items():
        state.hosts[name]["utilisation"] = utilisation
        state.hosts[name]["active"] = utilisation > 0.0
        assigned_cpu += utilisation * state.hosts[name]["cpu_capacity"]
    state.cooling_factor = normalised_cooling
    state.battery_command_kw = normalised_battery

    active_hosts = [
        name for name, host in state.hosts.items() if host["active"]
    ]
    return (
        f"ACCEPTED: {assigned_cpu:.2f} CPU units assigned; active hosts "
        f"{active_hosts}; cooling factor {state.cooling_factor:.2f}; "
        f"battery {state.battery_command_kw:.2f} kW. Finish now."
    )


# Fine-grained tools retained only for manual diagnostics.
@tool("Schedule one host")
def schedule_task(host_name: str, utilisation: float) -> str:
    """Set one host utilisation manually."""
    if host_name not in state.hosts:
        return f"ERROR: {host_name} does not exist."
    utilisation = float(utilisation)
    if not 0.0 <= utilisation <= 1.0:
        return "ERROR: utilisation must be between 0 and 1."
    state.hosts[host_name]["utilisation"] = utilisation
    return f"{host_name} utilisation set to {utilisation:.4f}."


@tool("Set one host power")
def set_host_power(host_name: str, active: bool) -> str:
    """Set one host power state manually."""
    if host_name not in state.hosts:
        return f"ERROR: {host_name} does not exist."
    host = state.hosts[host_name]
    if not active and host["utilisation"] > 0.0:
        return f"REJECTED: {host_name} still has workload."
    host["active"] = bool(active)
    return f"{host_name} power state set to {host['active']}."
