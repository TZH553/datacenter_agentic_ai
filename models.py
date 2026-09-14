from state import state


# =========================================================
# SERVER POWER
# =========================================================

def calculate_host_power(host):

    if not host["active"]:
        return 0.0

    utilisation = host["utilisation"]

    return (
        host["p_idle"]
        +
        utilisation
        * (
            host["p_max"]
            - host["p_idle"]
        )
    )


def calculate_it_power():

    state.it_power_kw = sum(
        calculate_host_power(host)
        for host in state.hosts.values()
    )

    return state.it_power_kw


# =========================================================
# COOLING
# =========================================================

def calculate_cooling_power():

    state.cooling_power_kw = (
        state.it_power_kw
        / state.cooling_cop
        * state.cooling_factor
    )

    return state.cooling_power_kw


def update_temperature():

    thermal_gain = 0.035
    thermal_decay = 0.10

    heat_effect = (
        state.it_power_kw
        * thermal_gain
    )

    cooling_effect = (
        state.cooling_power_kw
        * thermal_gain
    )

    ambient_effect = (
        state.ambient_temperature
        - state.temperature
    ) * thermal_decay

    state.temperature += (
        ambient_effect
        + heat_effect
        - cooling_effect
    )

    return state.temperature


# =========================================================
# BATTERY
# =========================================================
def update_battery(dt=1.0):

    command = state.battery_command_kw
    capacity = state.battery_capacity_kwh

    # ==========================================
    # DISCHARGE
    # ==========================================

    if command > 0:

        requested = min(
            command,
            state.battery_max_discharge_kw
        )

        # Battery must not discharge more power
        # than the data centre actually needs.
        residual_demand = max(
            0.0,
            state.total_power_kw - state.solar_kw
        )

        requested = min(
            requested,
            residual_demand
        )

        # Minimum allowed SOC
        min_soc = 0.10

        usable_energy = max(
            0.0,
            (state.battery_soc - min_soc)
            * capacity
        )

        maximum_possible_power = (
            usable_energy
            * state.battery_discharge_efficiency
            / dt
        )

        actual_power = min(
            requested,
            maximum_possible_power
        )

        energy_removed = (
            actual_power
            * dt
            / state.battery_discharge_efficiency
        )

        state.battery_soc -= (
            energy_removed / capacity
        )

        state.battery_power_kw = (
            actual_power
        )

    # ==========================================
    # CHARGE
    # ==========================================

    elif command < 0:

        requested = min(
            abs(command),
            state.battery_max_charge_kw
        )

        max_soc = 0.95

        available_space = max(
            0.0,
            (max_soc - state.battery_soc)
            * capacity
        )

        maximum_charge_power = (
            available_space
            / state.battery_charge_efficiency
            / dt
        )

        actual_power = min(
            requested,
            maximum_charge_power
        )

        stored_energy = (
            actual_power
            * dt
            * state.battery_charge_efficiency
        )

        state.battery_soc += (
            stored_energy / capacity
        )

        # Negative means charging
        state.battery_power_kw = (
            -actual_power
        )

    # ==========================================
    # IDLE
    # ==========================================

    else:

        state.battery_power_kw = 0.0


    state.battery_soc = max(
        0.10,
        min(
            0.95,
            state.battery_soc
        )
    )

    return state.battery_power_kw


# =========================================================
# GRID POWER
# =========================================================

def calculate_grid_power():

    state.total_power_kw = (
        state.it_power_kw
        +
        state.cooling_power_kw
    )

    # Energy balance:
    #
    # grid + solar + battery
    # =
    # IT + cooling
    #
    # battery positive = discharge

    state.grid_power_kw = max(
        0.0,

        state.total_power_kw
        - state.solar_kw
        - state.battery_power_kw
    )

    return state.grid_power_kw


# =========================================================
# COST
# =========================================================

def calculate_cost(dt=1.0):

    state.cost = (
        state.grid_power_kw
        * state.grid_price
        * dt
    )

    return state.cost