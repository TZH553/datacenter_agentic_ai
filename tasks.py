from crewai import Task

from agents import (
    scheduler_agent,
    power_governor_agent,
    cooling_agent,
    energy_agent
)


# =========================================================
# SCHEDULER TASK
# =========================================================

scheduler_task = Task(

    description="""
    Examine the current data-centre workload and host telemetry.

    Distribute the pending workload across active hosts.

    Requirements:

    1. No host should exceed 90% utilisation if avoidable.
    2. Consolidate workload onto fewer servers when possible.
    3. Leave unnecessary hosts at zero utilisation so the
       power governor can shut them down.
    4. Maintain enough active capacity for the current workload.
    5. Use the Schedule workload tool to implement decisions.

    You must inspect telemetry before making decisions.
    """,

    expected_output=(
        "A summary of workload placement including which "
        "hosts were assigned workload and their final "
        "utilisation."
    ),

    agent=scheduler_agent
)


# =========================================================
# POWER TASK
# =========================================================

power_task = Task(

    description="""
    Review the server state after workload scheduling.

    Identify hosts that have zero or effectively zero workload.

    Power off unnecessary idle hosts where safe.

    Requirements:

    1. Never switch off a host that still has workload.
    2. Maintain enough active hosts to serve the current load.
    3. Avoid unnecessary ON/OFF switching.
    4. Use Set host power only when a change is justified.
    """,

    expected_output=(
        "A summary of which hosts were kept active or "
        "powered off and why."
    ),

    agent=power_governor_agent,

    context=[
        scheduler_task
    ]
)


# =========================================================
# COOLING TASK
# =========================================================

cooling_task = Task(

    description="""
    Inspect current data-centre telemetry after workload and
    host-power decisions.

    Control the cooling system.

    Target operating range:

        19 C <= temperature <= 27 C

    Guidance:

    - If temperature is below 22 C, cooling may be reduced.
    - Around 22-24 C, normal cooling is normally appropriate.
    - Above 24 C, consider increasing cooling.
    - Above 26 C, cooling should be increased aggressively.

    Cooling factor constraints:

        minimum = 0.8
        normal  = 1.0
        maximum = 1.5

    Minimize cooling energy while preventing temperature
    violations.

    Use Set cooling level if a change is required.
    """,

    expected_output=(
        "The selected cooling factor and an explanation "
        "based on temperature and IT load."
    ),

    agent=cooling_agent,

    context=[
        scheduler_task,
        power_task
    ]
)


# =========================================================
# ENERGY TASK
# =========================================================

energy_task = Task(

    description="""
    Inspect current electricity price, solar generation,
    data-centre demand and battery SOC.

    Decide whether the battery should:

        CHARGE,
        DISCHARGE,
        or remain IDLE.

    Policy guidance:

    LOW electricity price:
        below $0.20/kWh

    NORMAL electricity price:
        $0.20-$0.35/kWh

    HIGH electricity price:
        above $0.35/kWh

    Battery constraints:

        minimum preferred SOC = 20%
        maximum preferred SOC = 90%

    Strategy:

    - Prefer solar power whenever available.
    - During high electricity prices, consider battery
      discharge if SOC is sufficient.
    - During low electricity prices or solar surplus,
      consider charging.
    - Avoid completely draining the battery.
    - Preserve some capacity for future disturbances.
    - Maximum charge/discharge power is 60 kW.

    Use Dispatch battery to implement the decision.
    """,

    expected_output=(
        "Battery operating mode, requested battery power "
        "and justification using price, solar generation "
        "and battery SOC."
    ),

    agent=energy_agent,

    context=[
        scheduler_task,
        power_task,
        cooling_task
    ]
)