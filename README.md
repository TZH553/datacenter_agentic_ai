# Data-Centre Agentic AI Optimisation Simulator

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
