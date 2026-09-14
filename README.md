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
