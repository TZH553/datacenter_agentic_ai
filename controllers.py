from models import estimate_cooling
from state import state


def _allocate_workload(target_utilisation):
    """Allocate explicit CPU units across heterogeneous clusters.

    Larger clusters are filled first to consolidate demand. The preferred
    utilisation is used first; capacity up to 100% is used only when total
    demand cannot otherwise be served.
    """
    target = max(0.01, min(1.0, float(target_utilisation)))
    ordered_hosts = sorted(
        state.hosts,
        key=lambda name: state.hosts[name]["cpu_capacity"],
        reverse=True,
    )

    utilisation = {name: 0.0 for name in state.hosts}
    power = {name: False for name in state.hosts}
    remaining_cpu = state.pending_workload_cpu

    # Preferred operating region.
    for name in ordered_hosts:
        if remaining_cpu <= 1e-9:
            break
        capacity = state.hosts[name]["cpu_capacity"]
        assigned_cpu = min(remaining_cpu, capacity * target)
        utilisation[name] = assigned_cpu / capacity
        power[name] = True
        remaining_cpu -= assigned_cpu

    # Use headroom up to 100% if demand exceeds preferred capacity.
    if remaining_cpu > 1e-9:
        for name in ordered_hosts:
            capacity = state.hosts[name]["cpu_capacity"]
            current_cpu = utilisation[name] * capacity
            headroom_cpu = capacity - current_cpu
            extra_cpu = min(remaining_cpu, headroom_cpu)
            utilisation[name] += extra_cpu / capacity
            power[name] = True
            remaining_cpu -= extra_cpu
            if remaining_cpu <= 1e-9:
                break

    return utilisation, power


def _predict_temperature(it_power_kw, heat_removed_kw):
    dt = state.timestep_h
    server_heat_kw = it_power_kw * state.server_heat_fraction
    envelope_heat_kw = (
        state.ambient_temperature - state.temperature
    ) * state.envelope_heat_transfer_kw_per_c
    net_thermal_energy_kwh = (
        server_heat_kw + envelope_heat_kw - heat_removed_kw
    ) * dt
    return (
        state.temperature
        + net_thermal_energy_kwh / state.thermal_mass_kwh_per_c
    )


class RuleBasedController:
    name = "Rule-Based EMS"

    def decide(self):
        utilisation, power = _allocate_workload(target_utilisation=0.90)

        if state.temperature > 26:
            cooling_factor = 1.40
        elif state.temperature > 24:
            cooling_factor = 1.20
        elif state.temperature < 22:
            cooling_factor = 0.90
        else:
            cooling_factor = 1.00

        if state.grid_price > 0.35 and state.battery_soc > 0.30:
            battery_command_kw = 15.0
        elif state.grid_price < 0.18 and state.battery_soc < 0.85:
            battery_command_kw = -10.0
        else:
            battery_command_kw = 0.0

        return {
            "host_utilisation": utilisation,
            "host_power": power,
            "cooling_factor": cooling_factor,
            "battery_command_kw": battery_command_kw,
        }


class FixedOptimisationController:
    name = "Fixed Optimisation"

    def __init__(self):
        self.cooling_options = [0.8, 1.0, 1.2, 1.4]
        self.battery_options = [-20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
        self.battery_degradation_cost_per_kwh = 0.04
        self.low_soc_threshold = 0.30
        self.low_soc_penalty_weight = 5.0
        self.temperature_penalty_weight = 100.0
        self.switching_penalty_weight = 0.05

    def decide(self):
        utilisation, power = _allocate_workload(target_utilisation=0.90)
        dt = state.timestep_h

        predicted_it_power_kw = sum(
            host["p_idle"]
            + utilisation[name] * (host["p_max"] - host["p_idle"])
            for name, host in state.hosts.items()
            if power[name]
        )
        switching_count = sum(
            state.hosts[name]["active"] != power[name] for name in state.hosts
        )

        best_score = float("inf")
        best_action = None

        for cooling_factor in self.cooling_options:
            (
                cooling_power_kw,
                heat_removed_kw,
                _,
            ) = estimate_cooling(
                predicted_it_power_kw,
                cooling_factor,
            )
            facility_power_kw = predicted_it_power_kw + cooling_power_kw
            predicted_temp = _predict_temperature(
                predicted_it_power_kw, heat_removed_kw
            )

            temperature_penalty = 0.0
            if predicted_temp > state.max_temp_c:
                temperature_penalty = (
                    predicted_temp - state.max_temp_c
                ) * self.temperature_penalty_weight
            elif predicted_temp < state.min_temp_c:
                temperature_penalty = (
                    state.min_temp_c - predicted_temp
                ) * self.temperature_penalty_weight

            for command_kw in self.battery_options:
                battery_power_kw = 0.0
                predicted_soc = state.battery_soc

                if command_kw > 0:
                    residual_kw = max(
                        0.0, facility_power_kw - state.solar_kw
                    )
                    usable_kwh = max(
                        0.0,
                        (state.battery_soc - state.battery_min_soc)
                        * state.battery_capacity_kwh,
                    )
                    battery_power_kw = min(
                        command_kw,
                        state.battery_max_discharge_kw,
                        residual_kw,
                        usable_kwh
                        * state.battery_discharge_efficiency
                        / dt,
                    )
                    predicted_soc -= (
                        battery_power_kw
                        * dt
                        / state.battery_discharge_efficiency
                        / state.battery_capacity_kwh
                    )

                elif command_kw < 0:
                    available_kwh = max(
                        0.0,
                        (state.battery_max_soc - state.battery_soc)
                        * state.battery_capacity_kwh,
                    )
                    charge_kw = min(
                        abs(command_kw),
                        state.battery_max_charge_kw,
                        available_kwh
                        / state.battery_charge_efficiency
                        / dt,
                    )
                    battery_power_kw = -charge_kw
                    predicted_soc += (
                        charge_kw
                        * dt
                        * state.battery_charge_efficiency
                        / state.battery_capacity_kwh
                    )

                grid_power_kw = max(
                    0.0,
                    facility_power_kw
                    - state.solar_kw
                    - battery_power_kw,
                )
                grid_cost = grid_power_kw * state.grid_price * dt
                degradation_cost = (
                    abs(battery_power_kw)
                    * dt
                    * self.battery_degradation_cost_per_kwh
                )
                low_soc_penalty = max(
                    0.0, self.low_soc_threshold - predicted_soc
                ) * self.low_soc_penalty_weight
                switching_penalty = (
                    switching_count * self.switching_penalty_weight
                )
                score = (
                    grid_cost
                    + degradation_cost
                    + low_soc_penalty
                    + temperature_penalty
                    + switching_penalty
                )

                if score < best_score:
                    best_score = score
                    best_action = {
                        "host_utilisation": utilisation.copy(),
                        "host_power": power.copy(),
                        "cooling_factor": cooling_factor,
                        "battery_command_kw": command_kw,
                    }

        return best_action


class RLController:
    name = "RL Controller"

    def decide(self):
        target = (
            0.80 if state.pending_workload_fraction > 0.80 else 0.70
        )
        utilisation, power = _allocate_workload(target)

        battery_command_kw = 0.0
        if state.grid_price > 0.35 and state.battery_soc > 0.35:
            battery_command_kw = 50.0
        elif state.solar_kw > 30 and state.battery_soc < 0.80:
            battery_command_kw = -30.0

        cooling_factor = 1.25 if state.temperature > 25 else 1.0

        return {
            "host_utilisation": utilisation,
            "host_power": power,
            "cooling_factor": cooling_factor,
            "battery_command_kw": battery_command_kw,
        }
