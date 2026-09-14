from crewai import Agent, LLM

from tools import (
    get_cluster_telemetry,
    schedule_task,
    set_host_power,
    set_cooling_level,
    dispatch_battery
)

#  model="lm_studio/qwen2.5-7b-instruct",

llm = LLM(
    model="openai/qwen2.5-7b-instruct",
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    temperature=0.1,
    stream=False
)

# =========================================================
# SCHEDULER AGENT
# =========================================================

scheduler_agent = Agent(

    role="Data Center Task Scheduler",

    goal=(
        "Assign the current workload across active server hosts "
        "while minimizing host overload, queue latency and "
        "unnecessary power consumption."
    ),

    backstory=(
        "You are an expert data-centre workload scheduler. "
        "You distribute CPU workload across server hosts and "
        "try to consolidate workloads when possible so unused "
        "servers may later be switched off."
    ),

    tools=[
        get_cluster_telemetry,
        schedule_task
    ],

    verbose=True,

    llm=llm
)


# =========================================================
# POWER GOVERNOR AGENT
# =========================================================

power_governor_agent = Agent(

    role="Server Power Governor",

    goal=(
        "Reduce unnecessary server power consumption by "
        "switching off genuinely idle hosts while preserving "
        "sufficient compute capacity."
    ),

    backstory=(
        "You manage server power states in a data centre. "
        "A host may only be switched off when its utilisation "
        "is effectively zero. Never switch off a server that "
        "still carries workload."
    ),

    tools=[
        get_cluster_telemetry,
        set_host_power
    ],

    verbose=True,

    llm=llm
)


# =========================================================
# COOLING AGENT
# =========================================================

cooling_agent = Agent(

    role="Data Center Cooling Controller",

    goal=(
        "Maintain server-room temperature between 19 and "
        "27 degrees Celsius while minimizing cooling energy."
    ),

    backstory=(
        "You are a thermal-management specialist responsible "
        "for data-centre HVAC operation. Excessive cooling "
        "wastes electricity, while insufficient cooling risks "
        "thermal violations and equipment damage. You adjust "
        "the cooling factor according to IT load and "
        "temperature."
    ),

    tools=[
        get_cluster_telemetry,
        set_cooling_level
    ],

    verbose=True,

    llm=llm
)


# =========================================================
# ENERGY AGENT
# =========================================================

energy_agent = Agent(

    role="Renewable Energy and Battery Manager",

    goal=(
        "Minimize grid electricity cost while maximizing "
        "solar-energy utilisation and maintaining sufficient "
        "battery reserve."
    ),

    backstory=(
        "You manage the data centre's battery energy storage "
        "system and renewable generation. You charge the "
        "battery during periods of renewable surplus or low "
        "electricity prices and discharge it when electricity "
        "is expensive, while protecting battery SOC limits."
    ),

    tools=[
        get_cluster_telemetry,
        dispatch_battery
    ],

    verbose=True,

    llm=llm
)