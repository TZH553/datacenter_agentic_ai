from crewai import Task

from agents import (
    compute_agent,
    cooling_agent,
    energy_agent,
    facility_energy_agent,
    integrated_ems_agent,
    power_governor_agent,
    scheduler_agent,
)


COMMON_LIMITS = """
Call Get cluster telemetry exactly once. Then call the required control tool
exactly once. All control inputs are simple numbers; never construct a host
dictionary or perform host-by-host calls. The control tool's returned text is
the authoritative result; do not restate, recalculate, or replace its values.
A control tool result completes the task.
Keep projected facility demand below the operating limit reported by
telemetry; this includes a safety margin below the physical power cap.
"""

COMPUTE_RULES = """
Choose a target_utilisation between 0.50 and 1.00. A lower target keeps more
hosts active with more headroom; a higher target consolidates demand and
usually reduces idle power. Prefer 0.90 unless current conditions justify
another value. Deterministic code calculates all individual allocations and
serves all feasible CPU demand.
Choose batch_service_fraction between 0.00 and 1.00. Interactive demand is
mandatory. Flexible batch work may be delayed during high price, low solar,
high temperature, or high power risk, but work at its deadline is forced by
the deterministic scheduler.
"""

COOLING_RULES = """
Keep temperature between 19.2 C and 22.8 C. Choose cooling_factor from 0.8 to
1.5 and cooling_setpoint_c from 19.2 to 22.8 C. Minimize cooling electricity
without causing a thermal violation. Higher setpoints improve COP but leave
less thermal safety margin. The setpoint is the actual thermostat target, so
do not select a value below the temperature needed for safe operation.
"""

ENERGY_RULES = """
Choose battery power in kW. Positive discharges and negative charges. Stay
within 10%-95% SOC and plus or minus 60 kW. Use solar surplus for charging. Grid charging is allowed only when price is
strictly below $0.18/kWh. At normal or high prices, choose zero or discharge;
never charge from the grid. Telemetry reports the amortized battery wear cost
per discharged kWh. Discharge only when grid price is higher than this
marginal wear cost. Consider discharge above $0.35/kWh when SOC is sufficient,
and preserve battery reserve. When price is above $0.35/kWh and battery energy
is cheaper than grid energy, explicitly request a positive discharge;
deterministic control will also replace a zero or charging request with safe
discharge when energy is available.
"""


scheduler_task = Task(
    description=COMMON_LIMITS + COMPUTE_RULES + """
Call Schedule all workload once with target_utilisation and
batch_service_fraction.
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
)

compute_task = Task(
    description=COMMON_LIMITS + COMPUTE_RULES + """
Call Apply complete compute plan once with target_utilisation and
batch_service_fraction.
""",
    expected_output="Accepted compute and power plan.",
    agent=compute_agent,
)

facility_energy_task = Task(
    description=COMMON_LIMITS + COOLING_RULES + ENERGY_RULES + """
Call Apply facility energy plan once with cooling_factor,
cooling_setpoint_c, and battery_power_kw. Keep projected facility power below
the 33.25 kW operating ceiling, which provides a 5% margin below the 35 kW
physical capacity.
""",
    expected_output="Accepted joint thermal and electrical plan.",
    agent=facility_energy_agent,
)

cooling_task_four = Task(
    description=COMMON_LIMITS + COOLING_RULES + """
Call Set cooling level once with cooling_factor and cooling_setpoint_c.
""",
    expected_output="Accepted cooling factor.",
    agent=cooling_agent,
)

cooling_task_three = Task(
    description=COMMON_LIMITS + COOLING_RULES + """
Call Set cooling level once with cooling_factor and cooling_setpoint_c.
""",
    expected_output="Accepted cooling factor.",
    agent=cooling_agent,
)

energy_task_four = Task(
    description=COMMON_LIMITS + ENERGY_RULES + """
Call Dispatch battery once with only power_kw.
""",
    expected_output="Accepted battery command.",
    agent=energy_agent,
)

energy_task_three = Task(
    description=COMMON_LIMITS + ENERGY_RULES + """
Call Dispatch battery once with only power_kw.
""",
    expected_output="Accepted battery command.",
    agent=energy_agent,
)

integrated_task = Task(
    description=COMMON_LIMITS + COMPUTE_RULES + COOLING_RULES + ENERGY_RULES + """
Call Apply complete EMS plan once with exactly five scalar arguments:
target_utilisation, cooling_factor, battery_power_kw,
batch_service_fraction, and cooling_setpoint_c.
""",
    expected_output="Accepted complete EMS plan.",
    agent=integrated_ems_agent,
)

idle_facility_task = Task(
    description=COMMON_LIMITS + COOLING_RULES + ENERGY_RULES + """
There is no incoming interactive work and no queued batch work. Compute hosts
have already been shut down deterministically, so do not schedule workload.
Call Apply facility energy plan once with cooling_factor,
cooling_setpoint_c, and battery_power_kw.
""",
    expected_output="Accepted idle-hour thermal and electrical plan.",
    agent=integrated_ems_agent,
)
