class DataCenterState:

    def __init__(self):

        self.hosts = {
            f"host_{i}": {
                "active": True,
                "utilisation": 0.0,
                "p_idle": 1.0,
                "p_max": 3.0,
            }
            for i in range(1, 11)
        }

        # Workload
        self.pending_workload = 0.0

        # Environment
        self.temperature = 22.0
        self.ambient_temperature = 24.0

        self.solar_kw = 0.0
        self.grid_price = 0.20

        # Battery
        self.battery_capacity_kwh = 300.0
        self.battery_soc = 0.70

        self.battery_max_charge_kw = 60.0
        self.battery_max_discharge_kw = 60.0

        self.battery_charge_efficiency = 0.95
        self.battery_discharge_efficiency = 0.95

        self.battery_command_kw = 0.0
        self.battery_power_kw = 0.0

        # Cooling
        self.cooling_factor = 1.0
        self.cooling_cop = 3.5

        # Calculated results
        self.it_power_kw = 0.0
        self.cooling_power_kw = 0.0
        self.total_power_kw = 0.0
        self.grid_power_kw = 0.0

        self.cost = 0.0


state = DataCenterState()


def reset_state():
    """
    Reset the shared data-centre state to identical
    initial conditions before every experiment.
    """

    new_state = DataCenterState()

    # Do NOT do:
    #
    # state = DataCenterState()
    #
    # because other modules may already have imported
    # the original state object.
    #
    # Instead, modify the existing object's attributes.

    state.__dict__.clear()

    state.__dict__.update(
        new_state.__dict__
    )