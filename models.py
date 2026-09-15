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


def estimate_cooling(
    it_power_kw,
    cooling_factor,
    temperature_c=None,
    ambient_temperature_c=None,
    cooling_setpoint_c=None,
):
    """Estimate cooling electricity and heat removal.

    Cooling factor controls available thermal-removal capacity rather than
    multiplying IT power. COP degrades as outdoor temperature rises, and fan
    power is incurred whenever cooling operates.
    """
    temperature_c = (
        state.temperature if temperature_c is None else temperature_c
    )
    ambient_temperature_c = (
        state.ambient_temperature
        if ambient_temperature_c is None
        else ambient_temperature_c
    )
    cooling_setpoint_c = (
        state.cooling_setpoint_c
        if cooling_setpoint_c is None
        else float(cooling_setpoint_c)
    )

    server_heat_kw = it_power_kw * state.server_heat_fraction
    envelope_heat_kw = (
        ambient_temperature_c - temperature_c
    ) * state.envelope_heat_transfer_kw_per_c

    # Do not deliberately cool below the configured minimum temperature.
    thermal_load_kw = max(0.0, server_heat_kw + envelope_heat_kw)
    if temperature_c <= state.min_temp_c:
        thermal_load_kw = min(
            thermal_load_kw,
            max(0.0, server_heat_kw + min(0.0, envelope_heat_kw)),
        )

    # Proportional thermostat action makes the chosen setpoint affect the
    # room temperature instead of merely changing cooling COP. At the
    # setpoint the plant removes the current heat load; above it, cooling is
    # increased so the room approaches the target over the configured time
    # constant. Below it, cooling is reduced and the room is allowed to warm.
    temperature_correction_kw = (
        (temperature_c - cooling_setpoint_c)
        * state.thermal_mass_kwh_per_c
        / state.thermal_control_time_constant_h
    )
    requested_heat_removal_kw = max(
        0.0, thermal_load_kw + temperature_correction_kw
    )

    available_capacity_kw = (
        state.cooling_nominal_capacity_kw * cooling_factor
    )
    heat_removed_kw = min(requested_heat_removal_kw, available_capacity_kw)

    # A lower supply-air setpoint and hotter outdoor air both make the
    # cooling plant work harder.
    ambient_penalty = max(0.0, ambient_temperature_c - 20.0)
    setpoint_penalty = max(0.0, 22.8 - cooling_setpoint_c)
    effective_cop = max(
        state.cooling_min_cop,
        state.cooling_cop
        - state.cooling_cop_temp_coefficient * ambient_penalty,
    )
    effective_cop = max(
        state.cooling_min_cop,
        effective_cop - 0.12 * setpoint_penalty,
    )

    fan_power_kw = (
        state.cooling_fan_power_kw * cooling_factor
        if heat_removed_kw > 0
        else 0.0
    )
    electrical_power_kw = heat_removed_kw / effective_cop + fan_power_kw
    return electrical_power_kw, heat_removed_kw, effective_cop


def calculate_cooling_power():
    """Calculate cooling power from thermal load and plant performance."""
    (
        state.cooling_power_kw,
        state.cooling_heat_removed_kw,
        state.effective_cooling_cop,
    ) = estimate_cooling(state.it_power_kw, state.cooling_factor)
    return state.cooling_power_kw


def update_temperature(dt=None):
    """Advance room temperature using a thermal energy balance."""
    dt = state.timestep_h if dt is None else float(dt)
    if dt <= 0:
        raise ValueError("dt must be positive")

    server_heat_kw = state.it_power_kw * state.server_heat_fraction
    envelope_heat_kw = (
        state.ambient_temperature - state.temperature
    ) * state.envelope_heat_transfer_kw_per_c
    net_thermal_energy_kwh = (
        server_heat_kw
        + envelope_heat_kw
        - state.cooling_heat_removed_kw
    ) * dt

    state.temperature += (
        net_thermal_energy_kwh / state.thermal_mass_kwh_per_c
    )
    return state.temperature


def update_battery(dt=None):
    """Apply battery command; positive discharges and negative charges."""
    dt = state.timestep_h if dt is None else float(dt)
    if dt <= 0:
        raise ValueError("dt must be positive")

    command_kw = float(state.battery_command_kw)
    capacity_kwh = state.battery_capacity_kwh

    if command_kw > 0:
        solar_to_load_kw = min(state.solar_kw, state.total_power_kw)
        residual_load_kw = max(
            0.0, state.total_power_kw - solar_to_load_kw
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
        state.battery_soc -= (
            actual_power_kw
            * dt
            / state.battery_discharge_efficiency
            / capacity_kwh
        )
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
        state.battery_soc += (
            actual_charge_kw
            * dt
            * state.battery_charge_efficiency
            / capacity_kwh
        )
        state.battery_power_kw = -actual_charge_kw

    else:
        state.battery_power_kw = 0.0

    state.battery_soc = max(
        state.battery_min_soc,
        min(state.battery_max_soc, state.battery_soc),
    )
    return state.battery_power_kw


def calculate_grid_power():
    """Route solar, battery, and grid power explicitly.

    Priority is solar-to-load, battery discharge-to-load, grid-to-load,
    solar surplus-to-battery, then grid-to-battery.
    """
    state.total_power_kw = state.it_power_kw + state.cooling_power_kw
    state.power_risk_ratio = (
        state.total_power_kw / state.facility_power_capacity_kw
    )
    charge_kw = max(0.0, -state.battery_power_kw)
    discharge_kw = max(0.0, state.battery_power_kw)

    state.solar_to_load_kw = min(state.solar_kw, state.total_power_kw)
    remaining_load_kw = state.total_power_kw - state.solar_to_load_kw

    battery_to_load_kw = min(discharge_kw, remaining_load_kw)
    state.grid_to_load_kw = max(
        0.0, remaining_load_kw - battery_to_load_kw
    )

    solar_surplus_kw = max(
        0.0, state.solar_kw - state.solar_to_load_kw
    )
    state.solar_to_battery_kw = min(charge_kw, solar_surplus_kw)
    state.grid_to_battery_kw = max(
        0.0, charge_kw - state.solar_to_battery_kw
    )
    state.solar_curtailed_kw = max(
        0.0, solar_surplus_kw - state.solar_to_battery_kw
    )

    state.grid_power_kw = (
        state.grid_to_load_kw + state.grid_to_battery_kw
    )
    return state.grid_power_kw


def calculate_cost(dt=None):
    """Calculate grid electricity cost for the current timestep."""
    dt = state.timestep_h if dt is None else float(dt)
    state.cost = state.grid_power_kw * state.grid_price * dt
    return state.cost
