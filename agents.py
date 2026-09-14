from crewai import Agent, LLM

from tools import (
    dispatch_battery,
    get_cluster_telemetry,
    schedule_task,
    set_cooling_level,
    set_host_power,
)


def _make_llm(max_tokens):
    return LLM(
        model="openai/qwen2.5-7b-instruct",
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        temperature=0.1,
        stream=False,
        max_tokens=max_tokens,
    )


specialist_llm = _make_llm(max_tokens=2048)
integrated_llm = _make_llm(max_tokens=4096)


scheduler_agent = Agent(
    role="Data Centre Task Scheduler",
    goal=(
        "Allocate all requested CPU units across clusters while avoiding "
        "overload and enabling idle clusters to be powered down."
    ),
    backstory=(
        "You specialize in heterogeneous CPU-capacity workload scheduling."
    ),
    tools=[get_cluster_telemetry, schedule_task],
    verbose=True,
    llm=specialist_llm,
)

power_governor_agent = Agent(
    role="Server Power Governor",
    goal=(
        "Minimize idle server power while preserving enough active CPU "
        "capacity for the allocated workload."
    ),
    backstory=(
        "You control cluster power states after workload placement."
    ),
    tools=[get_cluster_telemetry, set_host_power],
    verbose=True,
    llm=specialist_llm,
)

compute_agent = Agent(
    role="Integrated Compute Manager",
    goal=(
        "Jointly allocate CPU workload and choose cluster power states, "
        "serving all feasible demand with minimum server energy."
    ),
    backstory=(
        "You combine workload scheduling and power-state control so that "
        "migration and shutdown decisions cannot conflict."
    ),
    tools=[get_cluster_telemetry, schedule_task, set_host_power],
    verbose=True,
    llm=specialist_llm,
)

cooling_agent = Agent(
    role="Data Centre Thermal Manager",
    goal=(
        "Keep room temperature within its permitted range while minimizing "
        "cooling electricity."
    ),
    backstory=(
        "You manage cooling capacity using IT load, room temperature, "
        "ambient temperature, heat removal, and effective COP."
    ),
    tools=[get_cluster_telemetry, set_cooling_level],
    verbose=True,
    llm=specialist_llm,
)

energy_agent = Agent(
    role="Renewable Energy and Battery Manager",
    goal=(
        "Minimize grid cost, use available solar, and operate the battery "
        "within its SOC and power constraints."
    ),
    backstory=(
        "You coordinate solar, grid, and battery dispatch after compute and "
        "cooling decisions establish facility demand."
    ),
    tools=[get_cluster_telemetry, dispatch_battery],
    verbose=True,
    llm=specialist_llm,
)

integrated_ems_agent = Agent(
    role="Integrated Data Centre Energy Management Agent",
    goal=(
        "Jointly optimize CPU allocation, cluster power states, cooling, "
        "solar utilization, and battery dispatch while satisfying workload, "
        "temperature, battery, and equipment constraints."
    ),
    backstory=(
        "You are responsible for the complete data-centre control decision. "
        "You reason across compute, thermal, and electrical interactions "
        "instead of optimizing each subsystem independently."
    ),
    tools=[
        get_cluster_telemetry,
        schedule_task,
        set_host_power,
        set_cooling_level,
        dispatch_battery,
    ],
    verbose=True,
    llm=integrated_llm,
)
