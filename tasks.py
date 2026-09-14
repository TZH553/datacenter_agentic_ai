from crewai import Task

from agents import (
    compute_agent,
    cooling_agent,
    energy_agent,
    integrated_ems_agent,
    power_governor_agent,
    scheduler_agent,
)


COMMON_LIMITS = """
Call Get cluster telemetry exactly once. Then call the required control tool
exactly once with the complete decision. Do not call any tool host-by-host.
If the tool returns ACCEPTED, finish immediately without verification or
another tool call. Only correct and retry when the tool returns REJECTED.
"""

SCHEDULING_RULES = """
Demand is in CPU units. Assigned CPU equals utilisation times cpu_capacity.
Serve all feasible demand. Prefer at most 90% utilisation, but use up to 100%
when needed. Consolidate workload and use 0 for omitted hosts.
"""

COOLING_RULES = """
Keep temperature between 19 C and 27 C. Cooling factor is 0.8 to 1.5.
Minimize cooling electricity without causing a thermal violation.
"""

ENERGY_RULES = """
Positive battery power discharges; negative power charges. Stay within the
10%-95% SOC range and 60 kW power limits. Prefer solar, use discharge when
grid price is high, and preserve reserve.
"""


scheduler_task = Task(
    description=COMMON_LIMITS + SCHEDULING_RULES + """
Use Schedule all workload once. Provide one allocations object mapping host
names to utilisation fractions. Omitted hosts receive zero.
""",
    expected_output="Accepted batch CPU allocation.",
    agent=scheduler_agent,
)

power_task = Task(
    description=COMMON_LIMITS + """
Use Set all host power states once. Supply one active_hosts list containing
every loaded host and any additional host that must remain on.
""",
    expected_output="Accepted batch host-power plan.",
    agent=power_governor_agent,
    context=[scheduler_task],
)

compute_task = Task(
    description=COMMON_LIMITS + SCHEDULING_RULES + """
Use Apply complete compute plan once. It automatically turns allocated hosts
on and zero-utilisation hosts off.
""",
    expected_output="Accepted integrated compute plan.",
    agent=compute_agent,
)

cooling_task_four = Task(
    description=COMMON_LIMITS + COOLING_RULES + """
Use Set cooling level once after inspecting the compute outcome.
""",
    expected_output="Accepted cooling factor.",
    agent=cooling_agent,
    context=[scheduler_task, power_task],
)

cooling_task_three = Task(
    description=COMMON_LIMITS + COOLING_RULES + """
Use Set cooling level once after inspecting the compute outcome.
""",
    expected_output="Accepted cooling factor.",
    agent=cooling_agent,
    context=[compute_task],
)

energy_task_four = Task(
    description=COMMON_LIMITS + ENERGY_RULES + """
Use Dispatch battery once after compute and cooling are complete.
""",
    expected_output="Accepted battery command.",
    agent=energy_agent,
    context=[scheduler_task, power_task, cooling_task_four],
)

energy_task_three = Task(
    description=COMMON_LIMITS + ENERGY_RULES + """
Use Dispatch battery once after compute and cooling are complete.
""",
    expected_output="Accepted battery command.",
    agent=energy_agent,
    context=[compute_task, cooling_task_three],
)

integrated_task = Task(
    description=COMMON_LIMITS + SCHEDULING_RULES + COOLING_RULES + ENERGY_RULES + """
Use Apply complete EMS plan once. Supply the full allocations object, one
cooling_factor, and one battery_power_kw value. The tool validates and applies
the entire plan atomically.
""",
    expected_output="Accepted complete EMS plan.",
    agent=integrated_ems_agent,
)
