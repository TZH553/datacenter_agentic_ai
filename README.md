# Agentic-AI-Controlled General Cloud Data-Centre Simulator

This project models a **general heterogeneous cloud data centre controlled by
an Agentic AI energy-management system**. Agentic AI describes the controller;
it does not imply that every incoming workload is an AI workload. The trace
contains batch, interactive, MPI, and GPU-labelled cloud jobs.

This is a complete starter implementation of the architecture:

Workload
-> server clusters
-> IT power
-> heat
-> cooling
-> total electrical demand

Solar + Battery + Grid
-> energy balance
-> cost / emissions / peak demand

Agentic AI:
Monitor -> Predict -> Plan -> Optimise -> Decide -> Control
-> Simulator
-> new state
-> repeat

The four comparison systems are:

1. Rule-based EMS
2. Fixed optimisation
3. RL/ML baseline
4. Proposed Agentic AI

Run:

    pip install -r requirements.txt
    python main.py

Then:

    python experiment_rq1.py

The optimisation controller intentionally uses a small discrete action space.
This makes the project easy to debug and explain. It can later be replaced
with a formal Pyomo MILP while keeping the same simulator interface.

Important: the Agentic AI here is an explicit agent architecture with
monitoring, prediction, planning, optimisation and control modules. It is
not pretending that an LLM should calculate the physical equations itself.
An LLM can be connected later as the planning/orchestration layer.

## Configurable server hardware assumptions

The default homogeneous cluster is derived from explicit per-server variables
near the start of `config.py`:

```python
n_clusters = 10
hardware_profile_name = "General cloud CPU/GPU cluster"
processor_model = "Assumed 20-core server processor"
accelerator_model = "NVIDIA H100 SXM (assumed)"
servers_per_cluster = 5
processing_unit_name = "CPU-core equivalent"
processing_capacity_per_server = 20.0
server_idle_power_kw = 0.20
server_max_power_kw = 0.60
gpu_server_count = 5
gpus_per_gpu_server = 8
gpu_idle_power_kw = 0.07
gpu_max_power_kw = 0.70
```

Therefore, each default cluster contains five servers, provides 100 processing
units, consumes 1 kW while active and idle, and consumes 3 kW at full
utilisation. Across ten clusters, the model represents 50 servers and 1,000
processing units. These defaults preserve the earlier cluster-level values
while making their hardware basis explicit.

The CPU-server power values represent the server base system, including CPU,
memory, storage, fans and power-supply losses, but exclude accelerator power.
Accelerator power is calculated separately so GPU-labelled jobs do not make
ordinary CPU-only jobs consume GPU power. Five of the 50 servers are assumed
to be GPU-enabled, providing 40 GPUs in total. Change the model labels,
capacity and power values together when selecting real hardware. The optional
`p_idle_kw`, `p_max_kw`, and `cpu_capacity` fields override the derived
cluster values when required.

## Cloud workload trace time resolution

The supplied cloud workload CSV is an event/job trace and therefore has no
single native fixed timestep. Submission timestamps have one-minute
resolution, start and end timestamps have one-second resolution, and execution
times contain fractional seconds. The trace covers 1 January to 10 March 2024.

The simulator currently uses:

```python
timestep_h = 1.0
```

Consequently, jobs from the event trace are allocated across one-hour bins.
The one-hour value is a modelling choice, not the original CSV's sampling
interval. The current trace loader deliberately requires
`timestep_h = 1.0`. Supporting 15-minute operation would require changing
the trace aggregation and all solar, price and ambient profiles together.

The code now loads `cloud_workload_dataset.csv` from the same directory as
`experiment_rq1.py` and `main.py`. Each job enters the system in the
one-hour bin containing its `Submit_Time`. Its processing work is:

```text
CPU-core-hours = Used_CPUs × Execution_Time(Seconds) / 3600
```

GPU-labelled jobs retain that CPU work because accelerator jobs still require
host CPU resources. The source CSV has no requested-GPU field, so the model
uses the documented assumption:

```text
requested GPUs = Node_Count, for Job_Type = GPU
GPU-hours = requested GPUs × Execution_Time(Seconds) / 3600
```

Non-GPU jobs request zero GPUs. GPU work follows the same earliest-deadline-
first service order as its associated CPU work. GPU power is calculated from
the configured idle and maximum power after consolidating the selected GPU
work onto the fewest accelerators. The hourly files separately report CPU
server power, accelerator power, requested/assigned GPU demand, GPU
utilisation and GPU-hours. The experiment summary reports accelerator energy,
peak GPU demand and unmet GPU-hours.

These GPU values are simulation assumptions, not measurements contained in
the source trace. The model deliberately does not invent training/inference
labels, model size or token counts. Actual deployments should replace these
assumptions with scheduler requests and runtime accelerator telemetry.

Interactive jobs are mandatory in their arrival hour. Other jobs enter the
flexible queue with the following assumed deadlines:

| Job type | High priority | Medium priority | Low priority |
| --- | ---: | ---: | ---: |
| batch or unrecognised | 1 h | 4 h | 8 h |
| GPU | 2 h | 6 h | 12 h |
| MPI | 1 h | 3 h | 6 h |

These values are configurable in `config.py` through
`trace_batch_deadlines_h`, `trace_gpu_deadlines_h`, and
`trace_mpi_deadlines_h`. The tuple order is high, medium, low. Because the
source CSV contains no explicit deadline, these are documented simulation
assumptions.

By default the experiment starts at the earliest submission timestamp.
Set `trace_window_start` to an ISO timestamp to select another period.
`trace_scale_factor` can scale all CPU-core-hour values for capacity
sensitivity tests; its default is 1.0, which preserves the source data.

For an agentic run, the simulator skips the AI compute/scheduling task when an
hour contains no newly submitted jobs and the batch queue is also empty.
Hosts are shut down deterministically, while the relevant cooling and battery
agent tasks still run. Scheduling is not skipped when deferred work remains in
the queue, even if the current trace bin has no new submissions. Hourly output
records this decision in `scheduling_ai_skipped`, and the summary reports
`Scheduling AI Skips`.

The supervisory cooling decision is also reused when none of its relevant
inputs has changed materially since the last cooling decision. The comparison
includes room temperature, ambient temperature, CPU demand, GPU demand,
electricity price, solar power, battery SOC, and facility power-risk ratio.
The first hour always calls the cooling decision, and reuse is disabled when
room temperature is within the configured guard margin of either thermal
limit. The low-level thermal model continues to calculate cooling power and
room-temperature evolution every hour even when the AI decision is reused.

The deadbands are configurable near the cooling parameters in `config.py`:

```python
cooling_skip_temperature_tolerance_c = 0.10
cooling_skip_ambient_tolerance_c = 0.50
cooling_skip_cpu_tolerance = 0.10
cooling_skip_gpu_tolerance = 0.05
cooling_skip_price_tolerance = 0.005
cooling_skip_solar_tolerance_kw = 0.10
cooling_skip_battery_soc_tolerance = 0.01
cooling_skip_power_risk_tolerance = 0.02
cooling_skip_thermal_guard_margin_c = 0.50
```

When reuse is allowed, the previous `cooling_factor` and
`cooling_setpoint_c` are retained exactly. Other necessary agents can still
run: workload scheduling and battery dispatch are not skipped merely because
cooling is unchanged. Hourly output records `cooling_ai_skipped`, and the
experiment summary reports `Cooling AI Skips`.

## Heterogeneous cluster example

Input workload profiles remain normalized fractions from 0 to 1. The simulator
converts each value to CPU units using the installed capacities. To test clusters
with different compute and power characteristics:

```python
from config import ClusterConfig, Config

config = Config(
    clusters=(
        ClusterConfig(cpu_capacity=80, p_idle_kw=0.8, p_max_kw=2.4),
        ClusterConfig(cpu_capacity=120, p_idle_kw=1.1, p_max_kw=3.2),
        ClusterConfig(cpu_capacity=200, p_idle_kw=1.7, p_max_kw=5.0),
    )
)

results = run_simulation(
    controller=RuleBasedController(),
    workload_profile=workload,
    solar_profile=solar,
    price_profile=price,
    hours=len(workload),
    config=config,
)
```

For a workload fraction of 0.50, these clusters have 400 total CPU units and
therefore receive 200 CPU units of demand.

## Agent architecture comparison

The experiment compares the same physical simulation under three CrewAI
architectures:

- `four_agent`: scheduler, power governor, thermal manager, energy manager
- `three_agent`: combined compute manager, thermal manager, energy manager
- `single_agent`: one integrated EMS agent with all control tools

The specialist agents allow up to 2,048 output tokens per task. The integrated
single agent allows up to 4,096 output tokens for its larger decision.

Start LM Studio with the `qwen2.5-7b-instruct` model and its OpenAI-compatible
server at `http://127.0.0.1:1234/v1`. Then run a one-hour smoke test:

```bash
python experiment_rq1.py
```

For a longer final experiment on Windows Command Prompt:

```bat
set EXPERIMENT_HOURS=24
set AGENT_TIMEOUT_SECONDS=180
python experiment_rq1.py
```

For PowerShell:

```powershell
$env:EXPERIMENT_HOURS=24
$env:AGENT_TIMEOUT_SECONDS=180
python experiment_rq1.py
```

All architectures receive the same input profiles and each call to
`run_simulation` resets the physical state. Results are saved to separate
hourly CSV files and combined in `rq3_summary.csv`. The summary includes
energy, cost, PUE, temperature, unmet workload, response time, timeout,
fallback, and token-use metrics.
# Literature-grounded simulation extensions

The simulator now includes:

- dynamic room temperature with ambient-temperature and thermal-mass effects;
- cooling-capacity and 19.2-22.8 C setpoint control with setpoint/ambient-dependent COP;
- interactive demand plus deadline-constrained, deferrable batch workloads;
- a deterministic facility power cap with normal, warning, and critical risk levels;
- `two_agent`, where the Scheduler and Consolidation Manager jointly performs
  workload placement, batch shifting, consolidation, and host power control,
  while the Thermal and Electrical Energy Manager coordinates cooling and the
  battery.

`experiment_rq1.py` evaluates `four_agent`, `three_agent`, `two_agent`, and
`single_agent` configurations alongside the non-agentic baselines. It writes
two-agent hourly output to `rq3_agentic_two_hourly.csv` and includes deadline,
power-risk, power-cap, ambient-temperature, cooling-setpoint, and COP metrics.

## Battery lifecycle cost

Battery discharge is not treated as free energy. The simulation amortizes the
upfront battery cost over its expected full-cycle life:

```text
usable delivered energy per cycle
    = capacity × (maximum SOC - minimum SOC) × discharge efficiency

battery wear cost per discharged kWh
    = upfront battery cost × (1 - residual value fraction)
      / (cycle life × usable delivered energy per cycle)

total operating cost
    = grid electricity cost + battery degradation cost

battery net saving versus grid
    = counterfactual grid cost without storage - total operating cost
```

With the default 300 kWh battery, $400/kWh CAPEX, 10%-95% SOC window, 95%
discharge efficiency, and zero residual value, the wear cost is approximately
$0.04954 per discharged kWh for 10,000 cycles and $0.00991/kWh for 50,000
cycles. Grid electricity used to charge the battery is already included in
grid cost, so it is not added again at discharge.

Run the two cycle-life sensitivity cases in PowerShell:

```powershell
$env:BATTERY_CAPEX_PER_KWH=400
$env:BATTERY_CYCLE_LIFE=10000
python experiment_rq1.py

$env:BATTERY_CYCLE_LIFE=50000
python experiment_rq1.py
```

The hourly files report grid cost, battery wear cost, marginal battery wear
cost, and cumulative equivalent full cycles. The summary reports grid cost,
battery degradation cost, their combined total, the counterfactual cost of
buying the same load from the grid without storage, and the resulting net
battery saving. A positive net saving means storage was cheaper; a negative
value means buying from the grid would have been cheaper.
