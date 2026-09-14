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
exactly once. All control inputs are simple numbers; never construct a host
dictionary or perform host-by-host calls. If the tool returns ACCEPTED,
finish immediately. Retry only when it returns REJECTED.
"""

COMPUTE_RULES = """
Choose a target_utilisation between 0.50 and 1.00. A lower target keeps more
hosts active with more headroom; a higher target consolidates demand and
usually reduces idle power. Prefer 0.90 unless current conditions justify
another value. Deterministic code calculates all individual allocations and
serves all feasible CPU demand.
"""

COOLING_RULES = """
Keep temperature between 19 C and 27 C. Choose cooling_factor from 0.8 to
1.5. Minimize cooling electricity without causing a thermal violation.
"""

ENERGY_RULES = """
Choose battery power in kW. Positive discharges and negative charges. Stay
within 10%-95% SOC and plus or minus 60 kW. Prefer solar, consider discharge
at high grid price, and preserve battery reserve.
"""


scheduler_task = Task(
    description=COMMON_LIMITS + COMPUTE_RULES + """
Call Schedule all workload once with only target_utilisation.
""",
    expected_output="Accepted deterministic CPU allocation.",
    agent=scheduler_agent,
)

power_task = Task(
    description=COMMON_LIMITS + """
Call Set all host power states once with no arguments. It automatically keeps
loaded hosts on and turns zero-utilisation hosts off.
""",
    expected_output="Accepted deterministic host-power plan.",
    agent=power_governor_agent,
    context=[scheduler_task],
)

compute_task = Task(
    description=COMMON_LIMITS + COMPUTE_RULES + """
Call Apply complete compute plan once with only target_utilisation.
""",
    expected_output="Accepted compute and power plan.",
    agent=compute_agent,
)

cooling_task_four = Task(
    description=COMMON_LIMITS + COOLING_RULES + """
Call Set cooling level once with only cooling_factor.
""",
    expected_output="Accepted cooling factor.",
    agent=cooling_agent,
    context=[scheduler_task, power_task],
)

cooling_task_three = Task(
    description=COMMON_LIMITS + COOLING_RULES + """
Call Set cooling level once with only cooling_factor.
""",
    expected_output="Accepted cooling factor.",
    agent=cooling_agent,
    context=[compute_task],
)

energy_task_four = Task(
    description=COMMON_LIMITS + ENERGY_RULES + """
Call Dispatch battery once with only power_kw.
""",
    expected_output="Accepted battery command.",
    agent=energy_agent,
    context=[scheduler_task, power_task, cooling_task_four],
)

energy_task_three = Task(
    description=COMMON_LIMITS + ENERGY_RULES + """
Call Dispatch battery once with only power_kw.
""",
    expected_output="Accepted battery command.",
    agent=energy_agent,
    context=[compute_task, cooling_task_three],
)

integrated_task = Task(
    description=COMMON_LIMITS + COMPUTE_RULES + COOLING_RULES + ENERGY_RULES + """
Call Apply complete EMS plan once with exactly three scalar arguments:
target_utilisation, cooling_factor, and battery_power_kw.
""",
    expected_output="Accepted complete EMS plan.",
    agent=integrated_ems_agent,
)
