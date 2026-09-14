from config import ClusterConfig, Config


class DataCenterState:
    """Mutable physical state shared by controllers and simulation models."""

    def __init__(self, config=None):
        cfg = config or Config()

        self.timestep_h = cfg.timestep_h
        self.min_temp_c = cfg.min_temp_c
        self.max_temp_c = cfg.max_temp_c
        self.server_heat_fraction = cfg.server_heat_fraction
        self.thermal_mass_kwh_per_c = cfg.thermal_mass_kwh_per_c
        self.envelope_heat_transfer_kw_per_c = (
            cfg.envelope_heat_transfer_kw_per_c
        )

        cluster_specs = cfg.clusters
        if cluster_specs is None:
            cluster_specs = tuple(
                ClusterConfig(
                    cpu_capacity=cfg.cpu_capacity,
                    p_idle_kw=cfg.p_idle_kw,
                    p_max_kw=cfg.p_max_kw,
                )
                for _ in range(cfg.n_clusters)
            )

        if not cluster_specs:
            raise ValueError("At least one cluster must be configured.")

        self.hosts = {}
        for index, cluster in enumerate(cluster_specs, start=1):
            if cluster.cpu_capacity <= 0:
                raise ValueError("Cluster CPU capacity must be positive.")
            if not 0 <= cluster.p_idle_kw <= cluster.p_max_kw:
                raise ValueError(
                    "Cluster power must satisfy 0 <= p_idle_kw <= p_max_kw."
                )
            self.hosts[f"host_{index}"] = {
                "active": True,
                "utilisation": 0.0,
                "cpu_capacity": float(cluster.cpu_capacity),
                "p_idle": float(cluster.p_idle_kw),
                "p_max": float(cluster.p_max_kw),
            }

        # Input profiles are fractions, while scheduling uses explicit CPU units.
        self.pending_workload_fraction = 0.0
        self.pending_workload_cpu = 0.0

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
        self.cooling_min_cop = cfg.cooling_min_cop
        self.cooling_cop_temp_coefficient = (
            cfg.cooling_cop_temp_coefficient
        )
        self.cooling_nominal_capacity_kw = (
            cfg.cooling_nominal_capacity_kw
        )
        self.cooling_fan_power_kw = cfg.cooling_fan_power_kw
        self.effective_cooling_cop = cfg.cooling_cop
        self.cooling_heat_removed_kw = 0.0

        self.it_power_kw = 0.0
        self.cooling_power_kw = 0.0
        self.total_power_kw = 0.0
        self.grid_power_kw = 0.0
        self.solar_to_load_kw = 0.0
        self.solar_to_battery_kw = 0.0
        self.grid_to_load_kw = 0.0
        self.grid_to_battery_kw = 0.0
        self.solar_curtailed_kw = 0.0
        self.cost = 0.0

    @property
    def total_cpu_capacity(self):
        return sum(host["cpu_capacity"] for host in self.hosts.values())


state = DataCenterState()


def reset_state(config=None):
    """Reset the existing shared object so imported references remain valid."""
    new_state = DataCenterState(config)
    state.__dict__.clear()
    state.__dict__.update(new_state.__dict__)
