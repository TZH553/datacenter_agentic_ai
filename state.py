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
        self.thermal_control_time_constant_h = (
            cfg.thermal_control_time_constant_h
        )
        self.envelope_heat_transfer_kw_per_c = (
            cfg.envelope_heat_transfer_kw_per_c
        )

        self.hardware_profile_name = cfg.hardware_profile_name
        self.processor_model = cfg.processor_model
        self.accelerator_model = cfg.accelerator_model
        self.processing_unit_name = cfg.processing_unit_name
        self.servers_per_cluster = cfg.servers_per_cluster
        self.processing_capacity_per_server = (
            cfg.processing_capacity_per_server
        )
        self.server_idle_power_kw = cfg.server_idle_power_kw
        self.server_max_power_kw = cfg.server_max_power_kw

        if cfg.n_clusters <= 0:
            raise ValueError("Number of clusters must be positive.")
        if cfg.servers_per_cluster <= 0:
            raise ValueError("Servers per cluster must be positive.")
        if cfg.processing_capacity_per_server <= 0:
            raise ValueError(
                "Processing capacity per server must be positive."
            )
        if not 0 <= cfg.server_idle_power_kw <= cfg.server_max_power_kw:
            raise ValueError(
                "Server power must satisfy 0 <= idle <= maximum power."
            )

        cluster_specs = cfg.clusters
        uses_default_hardware_profile = cluster_specs is None
        if cluster_specs is None:
            derived_cpu_capacity = (
                cfg.servers_per_cluster
                * cfg.processing_capacity_per_server
            )
            derived_idle_kw = (
                cfg.servers_per_cluster * cfg.server_idle_power_kw
            )
            derived_max_kw = (
                cfg.servers_per_cluster * cfg.server_max_power_kw
            )
            cluster_specs = tuple(
                ClusterConfig(
                    cpu_capacity=(
                        derived_cpu_capacity
                        if cfg.cpu_capacity is None
                        else cfg.cpu_capacity
                    ),
                    p_idle_kw=(
                        derived_idle_kw
                        if cfg.p_idle_kw is None
                        else cfg.p_idle_kw
                    ),
                    p_max_kw=(
                        derived_max_kw
                        if cfg.p_max_kw is None
                        else cfg.p_max_kw
                    ),
                )
                for _ in range(cfg.n_clusters)
            )

        self.total_server_count = (
            cfg.n_clusters * cfg.servers_per_cluster
            if uses_default_hardware_profile
            else None
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
                "server_count": (
                    cfg.servers_per_cluster
                    if uses_default_hardware_profile
                    else None
                ),
            }

        # Input profiles are fractions, while scheduling uses explicit CPU units.
        self.pending_workload_fraction = 0.0
        self.pending_workload_cpu = 0.0
        self.interactive_workload_cpu = 0.0
        self.batch_arrival_cpu = 0.0
        self.batch_backlog = []
        self.batch_backlog_cpu = 0.0
        self.batch_served_cpu = 0.0
        self.batch_deadline_missed_cpu = 0.0
        self.interactive_workload_fraction = cfg.interactive_workload_fraction
        self.batch_deadline_h = cfg.batch_deadline_h

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
        if cfg.battery_capex_per_kwh < 0:
            raise ValueError("Battery CAPEX per kWh cannot be negative.")
        if cfg.battery_cycle_life <= 0:
            raise ValueError("Battery cycle life must be positive.")
        if not 0.0 <= cfg.battery_residual_value_fraction < 1.0:
            raise ValueError(
                "Battery residual value fraction must be in [0, 1)."
            )
        self.battery_capex_per_kwh = cfg.battery_capex_per_kwh
        self.battery_cycle_life = cfg.battery_cycle_life
        self.battery_residual_value_fraction = (
            cfg.battery_residual_value_fraction
        )
        self.battery_upfront_cost = (
            self.battery_capacity_kwh * self.battery_capex_per_kwh
        )
        self.battery_usable_cycle_kwh = (
            (self.battery_max_soc - self.battery_min_soc)
            * self.battery_capacity_kwh
            * self.battery_discharge_efficiency
        )
        amortisable_cost = self.battery_upfront_cost * (
            1.0 - self.battery_residual_value_fraction
        )
        lifetime_delivered_kwh = (
            self.battery_usable_cycle_kwh * self.battery_cycle_life
        )
        self.battery_degradation_cost_per_kwh = (
            amortisable_cost / lifetime_delivered_kwh
        )
        self.battery_equivalent_full_cycles = 0.0
        self.battery_command_kw = 0.0
        # Positive = discharge to load; negative = charging from supply.
        self.battery_power_kw = 0.0

        self.cooling_factor = 1.0
        self.cooling_setpoint_c = cfg.cooling_setpoint_c
        self.cooling_min_setpoint_c = cfg.cooling_min_setpoint_c
        self.cooling_max_setpoint_c = cfg.cooling_max_setpoint_c
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
        self.grid_energy_cost = 0.0
        self.battery_degradation_cost = 0.0
        self.counterfactual_grid_cost_without_battery = 0.0
        self.battery_net_saving_vs_grid = 0.0
        self.cost = 0.0
        self.facility_power_capacity_kw = cfg.facility_power_capacity_kw
        self.facility_operating_limit_kw = (
            cfg.facility_power_capacity_kw
            * cfg.facility_operating_limit_fraction
        )
        self.power_risk_warning_fraction = cfg.power_risk_warning_fraction
        self.power_risk_critical_fraction = cfg.power_risk_critical_fraction
        self.power_risk_ratio = 0.0

    @property
    def total_cpu_capacity(self):
        return sum(host["cpu_capacity"] for host in self.hosts.values())


state = DataCenterState()


def reset_state(config=None):
    """Reset the existing shared object so imported references remain valid."""
    new_state = DataCenterState(config)
    state.__dict__.clear()
    state.__dict__.update(new_state.__dict__)
