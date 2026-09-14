from config import Config


class DataCenterState:
    """Mutable physical state shared by controllers and simulation models."""

    def __init__(self, config=None):
        cfg = config or Config()

        self.timestep_h = cfg.timestep_h
        self.min_temp_c = cfg.min_temp_c
        self.max_temp_c = cfg.max_temp_c
        self.thermal_gain_c_per_kwh = cfg.thermal_gain_c_per_kwh
        self.thermal_decay_per_h = cfg.thermal_decay

        self.hosts = {
            f"host_{i}": {
                "active": True,
                "utilisation": 0.0,
                "cpu_capacity": cfg.cpu_capacity,
                "p_idle": cfg.p_idle_kw,
                "p_max": cfg.p_max_kw,
            }
            for i in range(1, cfg.n_clusters + 1)
        }

        # Workload is a fraction of total installed CPU capacity.
        self.pending_workload = 0.0

        self.temperature = cfg.initial_temp_c
        self.ambient_temperature = cfg.ambient_temp_c
        self.solar_kw = 0.0
        self.grid_price = 0.20

        self.battery_capacity_kwh = cfg.battery_capacity_kwh
        self.battery_soc = cfg.initial_soc
        self.battery_min_soc = 0.10
        self.battery_max_soc = 0.95
        self.battery_max_charge_kw = cfg.battery_max_charge_kw
        self.battery_max_discharge_kw = cfg.battery_max_discharge_kw
        self.battery_charge_efficiency = cfg.battery_charge_eff
        self.battery_discharge_efficiency = cfg.battery_discharge_eff
        self.battery_command_kw = 0.0
        # Positive = discharge to load; negative = charging from supply.
        self.battery_power_kw = 0.0

        self.cooling_factor = 1.0
        self.cooling_cop = cfg.cooling_cop

        self.it_power_kw = 0.0
        self.cooling_power_kw = 0.0
        self.total_power_kw = 0.0
        self.grid_power_kw = 0.0
        self.cost = 0.0


state = DataCenterState()


def reset_state():
    """Reset the existing shared object so imported references remain valid."""
    new_state = DataCenterState()
    state.__dict__.clear()
    state.__dict__.update(new_state.__dict__)
