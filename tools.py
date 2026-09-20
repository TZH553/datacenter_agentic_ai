from typing import Dict

from crewai.tools import tool

from models import calculate_host_power, estimate_cooling
from state import state
from workload import select_batch_service


def _build_allocations(
    target_utilisation: float,
    batch_service_fraction: float = 1.0,
) -> Dict[str, float]:
    """Deterministically allocate CPU demand across heterogeneous hosts."""
    target = float(target_utilisation)
    if not 0.50 <= target <= 1.0:
        raise ValueError(
            "Target utilisation must be between 0.50 and 1.00."
        )

    select_batch_service(batch_service_fraction)
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


def _normalise_setpoint(cooling_setpoint_c: float) -> float:
    value = float(cooling_setpoint_c)
    if not state.cooling_min_setpoint_c <= value <= state.cooling_max_setpoint_c:
        raise ValueError(
            f"Cooling setpoint must be between {state.cooling_min_setpoint_c} "
            f"and {state.cooling_max_setpoint_c} C."
        )
    return value


def _projected_facility_power(
    allocations: Dict[str, float],
    cooling_factor: float,
    cooling_setpoint_c: float | None = None,
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
        cooling_setpoint_c=cooling_setpoint_c,
    )
    return it_power_kw + cooling_power_kw


def _enforce_power_cap(
    allocations: Dict[str, float],
    cooling_factor: float,
    cooling_setpoint_c: float | None = None,
) -> float:
    projected_kw = _projected_facility_power(
        allocations, cooling_factor, cooling_setpoint_c
    )
    if projected_kw > state.facility_operating_limit_kw + 1e-9:
        raise ValueError(
            f"Projected facility power {projected_kw:.2f} kW exceeds "
            f"the {state.facility_operating_limit_kw:.2f} kW operating "
            f"ceiling (95% of the {state.facility_power_capacity_kw:.2f} "
            "kW physical cap). "
            "Reduce flexible batch service or cooling demand."
        )
    return projected_kw


def _commit_allocations(allocations: Dict[str, float]) -> float:
    """Apply a complete allocation so no utilisation can remain stale."""
    assigned_cpu = 0.0
    for name, utilisation in allocations.items():
        state.hosts[name]["utilisation"] = utilisation
        state.hosts[name]["active"] = utilisation > 0.0
        assigned_cpu += utilisation * state.hosts[name]["cpu_capacity"]
    return assigned_cpu


def _apply_safe_compute_fallback() -> tuple[float, float]:
    """Apply the largest safe deterministic workload allocation."""
    for service_fraction in (1.0, 0.75, 0.50, 0.25, 0.0):
        allocations = _build_allocations(1.0, service_fraction)
        try:
            _enforce_power_cap(
                allocations,
                state.cooling_factor,
                state.cooling_setpoint_c,
            )
        except ValueError:
            continue
        return _commit_allocations(allocations), service_fraction
    raise ValueError(
        "Mandatory interactive workload cannot fit below the operating ceiling."
    )


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

    # Do not consume cycle life when grid electricity is cheaper than the
    # amortized wear cost of delivering one kWh from the battery.
    if (
        value > 0
        and state.grid_price <= state.battery_degradation_cost_per_kwh
    ):
        value = 0.0

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
        if state.grid_price >= 0.18:
            # At normal/high prices, accept the plan but clamp charging
            # to available solar surplus. This avoids repeated agent retries.
            allowed_charge_kw = min(
                requested_charge_kw,
                solar_surplus_kw,
            )
            value = -allowed_charge_kw

    # A deterministic tariff guard prevents local LLMs from leaving stored
    # energy unused throughout expensive periods.
    if (
        state.grid_price > 0.35
        and state.grid_price > state.battery_degradation_cost_per_kwh
        and value <= 0.0
        and state.battery_soc > state.battery_min_soc + 0.05
    ):
        if allocations is None:
            allocations = {
                name: host["utilisation"]
                for name, host in state.hosts.items()
            }
        if cooling_factor is None:
            cooling_factor = state.cooling_factor
        residual_load_kw = max(
            0.0,
            _projected_facility_power(allocations, cooling_factor)
            - state.solar_kw,
        )
        usable_energy_kwh = (
            (state.battery_soc - state.battery_min_soc)
            * state.battery_capacity_kwh
        )
        energy_limited_kw = (
            usable_energy_kwh
            * state.battery_discharge_efficiency
            / state.timestep_h
        )
        value = min(
            50.0,
            residual_load_kw,
            state.battery_max_discharge_kw,
            energy_limited_kw,
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
            "interactive_workload_cpu_units": round(state.interactive_workload_cpu, 2),
            "batch_backlog_cpu_units": round(state.batch_backlog_cpu, 2),
            "nearest_batch_deadline_h": min(
                (job["hours_left"] for job in state.batch_backlog), default=None
            ),
            "temperature_C": round(state.temperature, 2),
            "temperature_range_C": [
                state.min_temp_c,
                state.max_temp_c,
            ],
            "IT_power_kw": round(state.it_power_kw, 2),
            "cooling_power_kw": round(state.cooling_power_kw, 2),
            "cooling_factor": state.cooling_factor,
            "cooling_setpoint_C": state.cooling_setpoint_c,
            "effective_cooling_cop": round(
                state.effective_cooling_cop, 2
            ),
            "solar_kw": round(state.solar_kw, 2),
            "grid_price": state.grid_price,
            "battery_SOC_percent": round(
                state.battery_soc * 100, 2
            ),
            "battery_command_kw": state.battery_command_kw,
            "battery_capex_USD": round(state.battery_upfront_cost, 2),
            "battery_cycle_life": state.battery_cycle_life,
            "battery_degradation_cost_per_kwh": round(
                state.battery_degradation_cost_per_kwh, 5
            ),
            "battery_saving_vs_grid_per_kwh": round(
                state.grid_price
                - state.battery_degradation_cost_per_kwh,
                5,
            ),
            "facility_power_capacity_kw": state.facility_power_capacity_kw,
            "facility_operating_limit_kw": state.facility_operating_limit_kw,
            "power_risk_ratio": round(state.power_risk_ratio, 4),
        }
    )


@tool("Schedule all workload")
def schedule_workload_batch(
    target_utilisation: float,
    batch_service_fraction: float = 1.0,
) -> str:
    """Allocate all CPU demand deterministically in one call.

    The agent chooses only a preferred target utilisation from 0.50 to 1.00.
    Deterministic code calculates every per-host allocation.
    """
    try:
        allocations = _build_allocations(
            target_utilisation, batch_service_fraction
        )
        _enforce_power_cap(
            allocations, state.cooling_factor, state.cooling_setpoint_c
        )
        assigned_cpu = 0.0
        for name, utilisation in allocations.items():
            state.hosts[name]["utilisation"] = utilisation
            assigned_cpu += utilisation * state.hosts[name]["cpu_capacity"]
    except (TypeError, ValueError) as error:
        try:
            assigned_cpu, service_fraction = _apply_safe_compute_fallback()
        except ValueError as fallback_error:
            return f"REJECTED: {error}; fallback failed: {fallback_error}"
        return (
            f"FALLBACK_APPLIED: requested plan was unsafe ({error}). "
            f"Assigned {assigned_cpu:.2f} CPU units with batch service "
            f"fraction {service_fraction:.2f}. Finish this task now."
        )

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
def apply_compute_plan(
    target_utilisation: float,
    batch_service_fraction: float = 1.0,
) -> str:
    """Allocate all CPU demand and derive every host power state."""
    try:
        allocations = _build_allocations(
            target_utilisation, batch_service_fraction
        )
        _enforce_power_cap(
            allocations, state.cooling_factor, state.cooling_setpoint_c
        )
        assigned_cpu = _commit_allocations(allocations)
    except (TypeError, ValueError) as error:
        try:
            assigned_cpu, service_fraction = _apply_safe_compute_fallback()
        except ValueError as fallback_error:
            return f"REJECTED: {error}; fallback failed: {fallback_error}"
        return (
            f"FALLBACK_APPLIED: requested plan was unsafe ({error}). "
            f"Assigned {assigned_cpu:.2f} CPU units with batch service "
            f"fraction {service_fraction:.2f}. Finish this task now."
        )

    active_hosts = [
        name for name, host in state.hosts.items() if host["active"]
    ]
    return (
        f"ACCEPTED: assigned {assigned_cpu:.2f} CPU units; active hosts "
        f"are {active_hosts}. Finish this task now."
    )


@tool("Set cooling level")
def set_cooling_level(
    cooling_factor: float,
    cooling_setpoint_c: float = 22.0,
) -> str:
    """Set relative cooling capacity from 0.8 to 1.5."""
    try:
        state.cooling_factor = _normalise_cooling(cooling_factor)
        state.cooling_setpoint_c = _normalise_setpoint(cooling_setpoint_c)
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"
    return (
        f"ACCEPTED: cooling factor is {state.cooling_factor:.2f}; "
        f"setpoint is {state.cooling_setpoint_c:.1f} C. "
        "Finish this task now."
    )


@tool("Apply facility energy plan")
def apply_facility_energy_plan(
    cooling_factor: float,
    cooling_setpoint_c: float,
    battery_power_kw: float,
) -> str:
    """Jointly control cooling and the battery for the two-agent model."""
    try:
        factor = _normalise_cooling(cooling_factor)
        setpoint = _normalise_setpoint(cooling_setpoint_c)
        allocations = {
            name: host["utilisation"] for name, host in state.hosts.items()
        }
        _enforce_power_cap(allocations, factor, setpoint)
        battery = _normalise_battery(battery_power_kw, cooling_factor=factor)
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"
    state.cooling_factor = factor
    state.cooling_setpoint_c = setpoint
    state.battery_command_kw = battery
    return (
        f"ACCEPTED: cooling factor {factor:.2f}, setpoint {setpoint:.1f} C, "
        f"battery {battery:.2f} kW. Finish now."
    )


@tool("Dispatch battery")
def dispatch_battery(power_kw: float) -> str:
    """Set battery power; positive discharges and negative charges."""
    requested_power_kw = float(power_kw)
    try:
        state.battery_command_kw = _normalise_battery(
            requested_power_kw
        )
    except (TypeError, ValueError) as error:
        return f"REJECTED: {error}"
    adjustment = ""
    if abs(state.battery_command_kw - requested_power_kw) > 1e-9:
        adjustment = (
            f" Requested {requested_power_kw:.2f} kW was safely adjusted "
            "to avoid normal/high-price grid charging."
        )
    return (
        f"ACCEPTED: battery command is "
        f"{state.battery_command_kw:.2f} kW.{adjustment} "
        "Finish this task now."
    )


@tool("Apply complete EMS plan")
def apply_ems_plan(
    target_utilisation: float,
    cooling_factor: float,
    battery_power_kw: float,
    batch_service_fraction: float = 1.0,
    cooling_setpoint_c: float = 22.0,
) -> str:
    """Apply compute, cooling, and battery decisions atomically.

    All inputs are simple scalars for reliable local-model tool calling.
    """
    requested_battery_kw = float(battery_power_kw)
    try:
        allocations = _build_allocations(
            target_utilisation, batch_service_fraction
        )
        normalised_cooling = _normalise_cooling(cooling_factor)
        normalised_setpoint = _normalise_setpoint(cooling_setpoint_c)
        _enforce_power_cap(
            allocations, normalised_cooling, normalised_setpoint
        )
        normalised_battery = _normalise_battery(
            requested_battery_kw,
            allocations=allocations,
            cooling_factor=normalised_cooling,
        )
    except (TypeError, ValueError) as error:
        try:
            assigned_cpu, service_fraction = _apply_safe_compute_fallback()
            normalised_cooling = state.cooling_factor
            normalised_setpoint = state.cooling_setpoint_c
            fallback_allocations = {
                name: host["utilisation"]
                for name, host in state.hosts.items()
            }
            normalised_battery = _normalise_battery(
                0.0,
                allocations=fallback_allocations,
                cooling_factor=normalised_cooling,
            )
        except ValueError as fallback_error:
            return f"REJECTED: {error}; fallback failed: {fallback_error}"
        state.battery_command_kw = normalised_battery
        return (
            f"FALLBACK_APPLIED: requested plan was unsafe ({error}). "
            f"Assigned {assigned_cpu:.2f} CPU units with batch service "
            f"fraction {service_fraction:.2f}, retained safe cooling, and "
            f"set battery to {normalised_battery:.2f} kW. Finish now."
        )

    assigned_cpu = _commit_allocations(allocations)
    state.cooling_factor = normalised_cooling
    state.cooling_setpoint_c = normalised_setpoint
    state.battery_command_kw = normalised_battery

    active_hosts = [
        name for name, host in state.hosts.items() if host["active"]
    ]
    adjustment = ""
    if abs(normalised_battery - requested_battery_kw) > 1e-9:
        adjustment = (
            f" Battery request {requested_battery_kw:.2f} kW was safely "
            "adjusted by the tariff and SOC safety policy."
        )
    return (
        f"ACCEPTED: {assigned_cpu:.2f} CPU units assigned; active hosts "
        f"{active_hosts}; cooling factor {state.cooling_factor:.2f}; "
        f"battery {state.battery_command_kw:.2f} kW.{adjustment} "
        "Finish now."
    )


# A successful control call is the task's authoritative final output. This
# prevents an extra LLM turn from delaying completion or inventing new values.
for _control_tool in (
    schedule_workload_batch,
    set_host_power_batch,
    apply_compute_plan,
    set_cooling_level,
    apply_facility_energy_plan,
    dispatch_battery,
    apply_ems_plan,
):
    setattr(_control_tool, "result_as_answer", True)


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
