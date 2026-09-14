from crewai import Task

from agents import (
    compute_agent,
    cooling_agent,
    energy_agent,
    integrated_ems_agent,
    power_governor_agent,
    scheduler_agent,
)


SCHEDULING_RULES = """
Inspect telemetry before acting. The pending demand is given in CPU units.
For each host, assigned CPU units equal utilisation times cpu_capacity.
Serve all feasible pending CPU demand. Prefer utilisation at or below 90%,
but use capacity up to 100% when necessary. Consolidate workload so unused
hosts can be switched off. Verify total assigned CPU units after acting.
"""

COOLING_RULES = """
Keep temperature between 19 C and 27 C. Cooling factor must be between 0.8
and 1.5. Use temperature, IT power, current heat removal, effective COP, and
ambient conditions. Minimize cooling electricity without causing a thermal
violation.
"""

ENERGY_RULES = """
Positive battery power discharges and negative power charges. Respect the
10%-95% hard SOC range and 60 kW charge/discharge limits. Prefer solar,
consider discharge when grid price is high, and preserve useful battery
reserve. Grid and solar source routing is calculated by deterministic code.
"""


scheduler_task = Task(
    description=SCHEDULING_RULES + """
Use Schedule workload to implement the allocation. Leave unnecessary hosts
at zero utilisation for the following power-governor task.
""",
    expected_output="Final workload allocation by host in CPU units and percent.",
    agent=scheduler_agent,
)

power_task = Task(
    description="""
Inspect the post-scheduling telemetry. Power off only hosts with effectively
zero workload. Keep every loaded host active and avoid unnecessary switching.
Use Set host power to implement justified changes.
""",
    expected_output="Final host power states and reasons for each change.",
    agent=power_governor_agent,
    context=[scheduler_task],
)

compute_task = Task(
    description=SCHEDULING_RULES + """
Use Schedule workload and Set host power to implement one consistent compute
decision. Never power off a loaded host. First ensure required hosts are on,
then allocate demand, then set unused hosts to zero and power them off.
""",
    expected_output=(
        "Final CPU allocation and power state for every host, with total "
        "assigned CPU units verified against demand."
    ),
    agent=compute_agent,
)

cooling_task_four = Task(
    description=COOLING_RULES + "
Inspect the results of both compute tasks.",
    expected_output="Selected cooling factor with thermal justification.",
    agent=cooling_agent,
    context=[scheduler_task, power_task],
)

cooling_task_three = Task(
    description=COOLING_RULES + "
Inspect the integrated compute decision.",
    expected_output="Selected cooling factor with thermal justification.",
    agent=cooling_agent,
    context=[compute_task],
)

energy_task_four = Task(
    description=ENERGY_RULES,
    expected_output="Battery command in kW with price, solar, load, and SOC justification.",
    agent=energy_agent,
    context=[scheduler_task, power_task, cooling_task_four],
)

energy_task_three = Task(
    description=ENERGY_RULES,
    expected_output="Battery command in kW with price, solar, load, and SOC justification.",
    agent=energy_agent,
    context=[compute_task, cooling_task_three],
)

integrated_task = Task(
    description=SCHEDULING_RULES + COOLING_RULES + ENERGY_RULES + """
Make one integrated decision in this exact order:
1. Read telemetry.
2. Allocate all feasible CPU demand.
3. Set safe cluster power states.
4. Select the cooling factor.
5. Select battery dispatch.
6. Read telemetry again and verify the final settings.
Use the tools to implement every decision; a textual recommendation alone
does not change the simulator.
""",
    expected_output=(
        "Verified final compute allocation, host power states, cooling "
        "factor, and battery command, including constraint checks."
    ),
    agent=integrated_ems_agent,
)
