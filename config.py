from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ClusterConfig:
    cpu_capacity: float
    p_idle_kw: float
    p_max_kw: float


@dataclass
class Config:
    hours: int = 168
    timestep_h: float = 1.0

    # Used to generate identical clusters when clusters is None.
    n_clusters: int = 10
    p_idle_kw: float = 1.0
    p_max_kw: float = 3.0
    cpu_capacity: float = 100.0

    # Provide explicit specifications to model heterogeneous clusters.
    clusters: Optional[Tuple[ClusterConfig, ...]] = None

    cooling_cop: float = 3.5
    cooling_min_cop: float = 1.5
    cooling_cop_temp_coefficient: float = 0.03
    cooling_nominal_capacity_kw: float = 25.0
    cooling_fan_power_kw: float = 0.50
    server_heat_fraction: float = 0.95
    thermal_mass_kwh_per_c: float = 80.0
    envelope_heat_transfer_kw_per_c: float = 0.50
    ambient_temp_c: float = 24.0
    initial_temp_c: float = 21.0
    min_temp_c: float = 19.0
    max_temp_c: float = 27.0

    battery_capacity_kwh: float = 300.0
    initial_soc: float = 0.70
    battery_max_charge_kw: float = 60.0
    battery_max_discharge_kw: float = 60.0
    battery_charge_eff: float = 0.95
    battery_discharge_eff: float = 0.95

    max_workload_shed: float = 0.20
    sla_limit: float = 0.05
    grid_emission_kg_per_kwh: float = 0.55
    random_seed: int = 42
