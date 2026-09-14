from state import state


def calculate_host_power(host):
    """Return host electrical power in kW."""
    if not host["active"]:
        return 0.0

    utilisation = max(0.0, min(1.0, float(host["utilisation"])))
    return host["p_idle"] + utilisation * (host["p_max"] - host["p_idle"])


def calculate_it_power():
    """Calculate total active-host IT power in kW."""
    state.it_power_kw = sum(
        calculate_host_power(host) for host in state.hosts.values()
    )
    return state.it_power_kw


def calculate_cooling_power():
    """Calculate cooling electrical power in kW.

    Base cooling power is IT heat load divided by COP. cooling_factor
    represents the controller's relative cooling effort.
    """
    state.cooling_power_kw = (
        state.it_power_kw / state.cooling_cop * state.cooling_factor
    )
    return state.cooling_power_kw


def update_temperature(dt=None):
    """Advance room temperature by one timestep.

    thermal_gain_c_per_kwh converts net thermal energy to temperature rise.
    thermal_decay_per_h is the hourly ambient coupling coefficient.
    """
    dt = state.timestep_h if dt is None else float(dt)

    ambient_effect = (
        state.ambient_temperature - state.temperature
    ) * state.thermal_decay_per_h * dt

    net_heat_kwh = (
        state.it_power_kw - state.cooling_power_kw
    ) * dt

    state.temperature += (
        ambient_effect
        + net_heat_kwh * state.thermal_gain_c_per_kwh
    )
    return state.temperature


def update_battery(dt=None):
    """Apply the battery command and update SOC.

    Positive battery power is discharge; negative battery power is charge.
    Power is kW, energy is power multiplied by dt in hours, and SOC is a
    dimensionless fraction.
    """
    dt = state.timestep_h if dt is None else float(dt)
    if dt <= 0:
        raise ValueError("dt must be positive")

    command_kw = float(state.battery_command_kw)
    capacity_kwh = state.battery_capacity_kwh

    if command_kw > 0:
        residual_load_kw = max(
            0.0, state.total_power_kw - state.solar_kw
        )
        usable_energy_kwh = max(
            0.0,
            (state.battery_soc - state.battery_min_soc) * capacity_kwh,
        )
        soc_limited_power_kw = (
            usable_energy_kwh * state.battery_discharge_efficiency / dt
        )

        actual_power_kw = min(
            command_kw,
            state.battery_max_discharge_kw,
            residual_load_kw,
            soc_limited_power_kw,
        )
        energy_removed_kwh = (
            actual_power_kw * dt / state.battery_discharge_efficiency
        )
        state.battery_soc -= energy_removed_kwh / capacity_kwh
        state.battery_power_kw = actual_power_kw

    elif command_kw < 0:
        available_space_kwh = max(
            0.0,
            (state.battery_max_soc - state.battery_soc) * capacity_kwh,
        )
        soc_limited_power_kw = (
            available_space_kwh / state.battery_charge_efficiency / dt
        )
        actual_charge_kw = min(
            abs(command_kw),
            state.battery_max_charge_kw,
            soc_limited_power_kw,
        )
        stored_energy_kwh = (
            actual_charge_kw * dt * state.battery_charge_efficiency
        )
        state.battery_soc += stored_energy_kwh / capacity_kwh
        state.battery_power_kw = -actual_charge_kw

    else:
        state.battery_power_kw = 0.0

    state.battery_soc = max(
        state.battery_min_soc,
        min(state.battery_max_soc, state.battery_soc),
    )
    return state.battery_power_kw


def calculate_grid_power():
    """Balance instantaneous power in kW.

    grid + solar + battery_discharge = facility + battery_charge.
    With signed battery_power_kw this becomes:
    grid = facility - solar - battery_power.
    """
    state.total_power_kw = state.it_power_kw + state.cooling_power_kw
    state.grid_power_kw = max(
        0.0,
        state.total_power_kw - state.solar_kw - state.battery_power_kw,
    )
    return state.grid_power_kw


def calculate_cost(dt=None):
    """Calculate grid electricity cost for the current timestep."""
    dt = state.timestep_h if dt is None else float(dt)
    state.cost = state.grid_power_kw * state.grid_price * dt
    return state.cost
