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

    # Default homogeneous hardware profile. Base-server power covers CPU,
    # memory, fans and storage; accelerator power is modelled separately.
    n_clusters: int = 10
    hardware_profile_name: str = "General cloud CPU/GPU cluster"
    processor_model: str = "Assumed 20-core server processor"
    accelerator_model: str = "NVIDIA H100 SXM (assumed)"
    servers_per_cluster: int = 5
    processing_unit_name: str = "CPU-core equivalent"
    processing_capacity_per_server: float = 20.0
    server_idle_power_kw: float = 0.20
    server_max_power_kw: float = 0.60

    # Accelerator assumptions apply only to trace rows labelled GPU. The
    # source trace does not report GPU counts, so Node_Count is interpreted as
    # the requested GPU count. CPU server power above excludes GPU power.
    gpu_server_count: int = 5
    gpus_per_gpu_server: int = 8
    gpu_idle_power_kw: float = 0.07
    gpu_max_power_kw: float = 0.70

    # Optional cluster-level overrides retained for sensitivity studies and
    # backwards compatibility. None derives each value from the server profile.
    p_idle_kw: Optional[float] = None
    p_max_kw: Optional[float] = None
    cpu_capacity: Optional[float] = None

    # Provide explicit specifications to model heterogeneous clusters.
    clusters: Optional[Tuple[ClusterConfig, ...]] = None

    cooling_cop: float = 3.5
    cooling_min_cop: float = 1.5
    cooling_cop_temp_coefficient: float = 0.03
    cooling_nominal_capacity_kw: float = 60.0
    cooling_fan_power_kw: float = 0.50
    cooling_setpoint_c: float = 22.0
    cooling_min_setpoint_c: float = 19.2
    cooling_max_setpoint_c: float = 22.8
    server_heat_fraction: float = 0.95
    thermal_mass_kwh_per_c: float = 80.0
    thermal_control_time_constant_h: float = 2.0
    envelope_heat_transfer_kw_per_c: float = 0.50
    ambient_temp_c: float = 24.0
    initial_temp_c: float = 21.0
    min_temp_c: float = 19.2
    max_temp_c: float = 22.8

    facility_power_capacity_kw: float = 75.0
    facility_operating_limit_fraction: float = 0.95
    power_risk_warning_fraction: float = 0.80
    power_risk_critical_fraction: float = 0.95

    interactive_workload_fraction: float = 0.70
    batch_deadline_h: int = 4

    # Cloud workload trace settings. Priority tuple order is
    # (high, medium, low). Interactive jobs are mandatory in their arrival bin.
    trace_workload_filename: str = "cloud_workload_dataset.csv"
    trace_scale_factor: float = 1.0
    trace_window_start: Optional[str] = None
    trace_batch_deadlines_h: Tuple[int, int, int] = (1, 4, 8)
    trace_gpu_deadlines_h: Tuple[int, int, int] = (2, 6, 12)
    trace_mpi_deadlines_h: Tuple[int, int, int] = (1, 3, 6)

    battery_capacity_kwh: float = 300.0
    initial_soc: float = 0.70
    battery_max_charge_kw: float = 60.0
    battery_max_discharge_kw: float = 60.0
    battery_charge_eff: float = 0.95
    battery_discharge_eff: float = 0.95
    # Lifecycle-cost assumptions. A full cycle uses the configured SOC window.
    # Set battery_cycle_life to 50_000 for the professor's alternative case.
    battery_capex_per_kwh: float = 400.0
    battery_cycle_life: int = 10_000
    battery_residual_value_fraction: float = 0.0

    max_workload_shed: float = 0.20
    sla_limit: float = 0.05
    grid_emission_kg_per_kwh: float = 0.55
    random_seed: int = 42
