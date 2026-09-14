import math

from state import state


class RuleBasedController:

    name = "Rule-Based EMS"

    def decide(self):

        action = {
            "host_utilisation": {},
            "host_power": {},
            "cooling_factor": 1.0,
            "battery_command_kw": 0.0
        }

        total_hosts = len(state.hosts)

        # pending_workload is assumed to be a fraction
        # of TOTAL data-centre capacity.
        total_required_utilisation = (
            state.pending_workload
            * total_hosts
        )

        max_utilisation = 0.90

        required_hosts = max(
            1,
            math.ceil(
                total_required_utilisation
                / max_utilisation
            )
        )

        host_names = list(
            state.hosts.keys()
        )

        remaining = total_required_utilisation

        for i, host_name in enumerate(host_names):

            if i < required_hosts:

                utilisation = min(
                    max_utilisation,
                    remaining
                )

                utilisation = max(
                    0.0,
                    utilisation
                )

                action["host_power"][
                    host_name
                ] = True

                action["host_utilisation"][
                    host_name
                ] = utilisation

                remaining -= utilisation

            else:

                action["host_utilisation"][
                    host_name
                ] = 0.0

                action["host_power"][
                    host_name
                ] = False


        # Cooling rule
        if state.temperature > 26:
            action["cooling_factor"] = 1.40

        elif state.temperature > 24:
            action["cooling_factor"] = 1.20

        elif state.temperature < 22:
            action["cooling_factor"] = 0.90

        else:
            action["cooling_factor"] = 1.00


        # Battery rule
        if (
            state.grid_price > 0.35
            and state.battery_soc > 0.30
        ):
            action["battery_command_kw"] = 15.0

        elif (
            state.grid_price < 0.18
            and state.battery_soc < 0.85
        ):
            action["battery_command_kw"] = -10.0

        else:
            action["battery_command_kw"] = 0.0


        return action


class FixedOptimisationController:

    name = "Fixed Optimisation"

    def __init__(self):

        # Candidate cooling values
        self.cooling_options = [
            0.8,
            1.0,
            1.2,
            1.4
        ]

        # Candidate battery commands
        self.battery_options = [
            -20.0,   # charge
            -10.0,
            0.0,
            10.0,
            20.0,
            30.0
        ]

        # Optimisation weights
        self.battery_degradation_cost = 0.04

        self.low_soc_threshold = 0.30

        self.low_soc_penalty_weight = 5.0

        self.temperature_penalty_weight = 100.0

        self.switching_penalty_weight = 0.05


    def decide(self):

        best_action = None

        best_score = float("inf")

        total_hosts = len(
            state.hosts
        )

        # ======================================
        # WORKLOAD REQUIREMENT
        # ======================================

        # Example:
        # pending_workload = 0.7
        # 10 hosts
        #
        # Total required host-equivalent load = 7.0

        total_required_utilisation = (
            state.pending_workload
            * total_hosts
        )

        max_utilisation = 0.90

        required_hosts = max(
            1,
            math.ceil(
                total_required_utilisation
                / max_utilisation
            )
        )

        required_hosts = min(
            required_hosts,
            total_hosts
        )

        host_names = list(
            state.hosts.keys()
        )


        # ======================================
        # BUILD SERVER ALLOCATION
        # ======================================

        base_host_utilisation = {}

        base_host_power = {}

        remaining = (
            total_required_utilisation
        )

        for i, host_name in enumerate(
            host_names
        ):

            if i < required_hosts:

                utilisation = min(
                    max_utilisation,
                    remaining
                )

                utilisation = max(
                    0.0,
                    utilisation
                )

                base_host_utilisation[
                    host_name
                ] = utilisation

                base_host_power[
                    host_name
                ] = True

                remaining -= utilisation

            else:

                base_host_utilisation[
                    host_name
                ] = 0.0

                base_host_power[
                    host_name
                ] = False


        # ======================================
        # ESTIMATE IT POWER
        # ======================================

        predicted_it_power = 0.0

        for host_name in host_names:

            utilisation = (
                base_host_utilisation[
                    host_name
                ]
            )

            active = (
                base_host_power[
                    host_name
                ]
            )

            if active:

                host = state.hosts[
                    host_name
                ]

                predicted_it_power += (
                    host["p_idle"]
                    + utilisation
                    * (
                        host["p_max"]
                        - host["p_idle"]
                    )
                )


        # ======================================
        # TEST CANDIDATE ACTIONS
        # ======================================

        for cooling_factor in (
            self.cooling_options
        ):

            predicted_cooling_power = (
                predicted_it_power
                / state.cooling_cop
                * cooling_factor
            )

            predicted_total_power = (
                predicted_it_power
                + predicted_cooling_power
            )


            for battery_command in (
                self.battery_options
            ):

                # ----------------------------------
                # Battery command filtering
                # ----------------------------------

                if (
                    battery_command > 0
                    and state.battery_soc
                    <= 0.20
                ):
                    continue

                if (
                    battery_command < 0
                    and state.battery_soc
                    >= 0.90
                ):
                    continue


                # ----------------------------------
                # Limit actual battery use
                # ----------------------------------

                residual_load = max(
                    0.0,
                    predicted_total_power
                    - state.solar_kw
                )

                predicted_battery_power = 0.0

                predicted_soc = (
                    state.battery_soc
                )


                # DISCHARGE
                if battery_command > 0:

                    predicted_battery_power = min(
                        battery_command,
                        state.battery_max_discharge_kw,
                        residual_load
                    )

                    energy_removed = (
                        predicted_battery_power
                        / state.battery_discharge_efficiency
                    )

                    predicted_soc -= (
                        energy_removed
                        / state.battery_capacity_kwh
                    )


                # CHARGE
                elif battery_command < 0:

                    charge_power = min(
                        abs(battery_command),
                        state.battery_max_charge_kw
                    )

                    predicted_battery_power = (
                        -charge_power
                    )

                    stored_energy = (
                        charge_power
                        * state.battery_charge_efficiency
                    )

                    predicted_soc += (
                        stored_energy
                        / state.battery_capacity_kwh
                    )


                # Hard SOC range
                if not (
                    0.10
                    <= predicted_soc
                    <= 0.95
                ):
                    continue


                # ----------------------------------
                # Grid power
                # ----------------------------------

                predicted_grid_power = max(
                    0.0,
                    predicted_total_power
                    - state.solar_kw
                    - predicted_battery_power
                )


                # ----------------------------------
                # Electricity cost
                # ----------------------------------

                predicted_grid_cost = (
                    predicted_grid_power
                    * state.grid_price
                )


                # ----------------------------------
                # Battery degradation cost
                # ----------------------------------

                battery_degradation_cost = (
                    max(
                        0.0,
                        predicted_battery_power
                    )
                    * self.battery_degradation_cost
                )


                # ----------------------------------
                # Low SOC penalty
                # ----------------------------------

                low_soc_penalty = 0.0

                if (
                    predicted_soc
                    < self.low_soc_threshold
                ):

                    low_soc_penalty = (
                        self.low_soc_threshold
                        - predicted_soc
                    ) * self.low_soc_penalty_weight


                # ----------------------------------
                # Approximate temperature prediction
                # ----------------------------------

                thermal_gain = 0.035

                thermal_decay = 0.10

                heat_effect = (
                    predicted_it_power
                    * thermal_gain
                )

                cooling_effect = (
                    predicted_cooling_power
                    * thermal_gain
                )

                ambient_effect = (
                    state.ambient_temperature
                    - state.temperature
                ) * thermal_decay

                predicted_temperature = (
                    state.temperature
                    + ambient_effect
                    + heat_effect
                    - cooling_effect
                )


                temperature_penalty = 0.0

                if predicted_temperature > 27:

                    temperature_penalty = (
                        predicted_temperature
                        - 27
                    ) * self.temperature_penalty_weight

                elif predicted_temperature < 19:

                    temperature_penalty = (
                        19
                        - predicted_temperature
                    ) * self.temperature_penalty_weight


                # ----------------------------------
                # Switching penalty
                # ----------------------------------

                switching_count = 0

                for host_name in host_names:

                    current_state = (
                        state.hosts[
                            host_name
                        ]["active"]
                    )

                    new_state = (
                        base_host_power[
                            host_name
                        ]
                    )

                    if current_state != new_state:

                        switching_count += 1


                switching_penalty = (
                    switching_count
                    * self.switching_penalty_weight
                )


                # ==================================
                # OBJECTIVE FUNCTION
                # ==================================

                score = (
                    predicted_grid_cost
                    + battery_degradation_cost
                    + low_soc_penalty
                    + temperature_penalty
                    + switching_penalty
                )


                if score < best_score:

                    best_score = score

                    best_action = {

                        "host_utilisation":
                            base_host_utilisation.copy(),

                        "host_power":
                            base_host_power.copy(),

                        "cooling_factor":
                            cooling_factor,

                        "battery_command_kw":
                            battery_command
                    }


        # ======================================
        # SAFE FALLBACK
        # ======================================

        if best_action is None:

            best_action = {

                "host_utilisation":
                    base_host_utilisation,

                "host_power":
                    base_host_power,

                "cooling_factor":
                    1.0,

                "battery_command_kw":
                    0.0
            }


        return best_action

# ============================================================
# 3. SIMPLE RL-STYLE CONTROLLER
# ============================================================

class RLController:

    name = "RL Controller"

    def decide(self):

        action = {
            "host_utilisation": {},
            "host_power": {},
            "cooling_factor": 1.0,
            "battery_command_kw": 0.0
        }


        # Example learned-style policy

        if state.pending_workload > 0.80:

            target_utilisation = 0.80

        else:

            target_utilisation = 0.70


        required_hosts = max(
            1,
            int(
                state.pending_workload
                /
                target_utilisation
            )
            + 1
        )


        remaining = state.pending_workload


        for i, host_name in enumerate(
            state.hosts.keys()
        ):

            if i < required_hosts:

                action[
                    "host_power"
                ][host_name] = True

                utilisation = min(
                    remaining,
                    target_utilisation
                )

                action[
                    "host_utilisation"
                ][host_name] = utilisation

                remaining -= utilisation

            else:

                action[
                    "host_utilisation"
                ][host_name] = 0.0

                action[
                    "host_power"
                ][host_name] = False


        # Battery policy

        if (
            state.grid_price > 0.35
            and state.battery_soc > 0.35
        ):

            action[
                "battery_command_kw"
            ] = 50


        elif (
            state.solar_kw > 30
            and state.battery_soc < 0.80
        ):

            action[
                "battery_command_kw"
            ] = -30


        # Cooling policy

        if state.temperature > 25:

            action[
                "cooling_factor"
            ] = 1.25


        return action