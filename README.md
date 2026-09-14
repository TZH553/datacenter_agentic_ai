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
