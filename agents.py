from crewai import Agent, LLM

from tools import (
    apply_compute_plan,
    apply_ems_plan,
    dispatch_battery,
    get_cluster_telemetry,
    schedule_workload_batch,
    set_cooling_level,
    set_host_power_batch,
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
    goal="Allocate all CPU demand with one batch scheduling call.",
    backstory="You specialize in heterogeneous CPU-capacity scheduling.",
    tools=[get_cluster_telemetry, schedule_workload_batch],
    verbose=True,
    max_iter=5,
    llm=specialist_llm,
)

power_governor_agent = Agent(
    role="Server Power Governor",
    goal="Set every cluster power state safely in one batch call.",
    backstory="You eliminate idle-server power after workload placement.",
    tools=[get_cluster_telemetry, set_host_power_batch],
    verbose=True,
    max_iter=5,
    llm=specialist_llm,
)

compute_agent = Agent(
    role="Integrated Compute Manager",
    goal="Apply one complete CPU allocation and power-state plan.",
    backstory="You jointly manage workload placement and cluster power.",
    tools=[get_cluster_telemetry, apply_compute_plan],
    verbose=True,
    max_iter=5,
    llm=specialist_llm,
)

cooling_agent = Agent(
    role="Data Centre Thermal Manager",
    goal="Select cooling capacity with one control call.",
    backstory="You minimize cooling energy within thermal constraints.",
    tools=[get_cluster_telemetry, set_cooling_level],
    verbose=True,
    max_iter=5,
    llm=specialist_llm,
)

energy_agent = Agent(
    role="Renewable Energy and Battery Manager",
    goal="Select battery dispatch with one control call.",
    backstory="You coordinate solar, grid price, and battery reserves.",
    tools=[get_cluster_telemetry, dispatch_battery],
    verbose=True,
    max_iter=5,
    llm=specialist_llm,
)

integrated_ems_agent = Agent(
    role="Integrated Data Centre Energy Management Agent",
    goal="Apply one complete compute, cooling, and battery plan.",
    backstory=(
        "You make one globally coordinated decision across compute, "
        "thermal, and electrical systems."
    ),
    tools=[get_cluster_telemetry, apply_ems_plan],
    verbose=True,
    max_iter=5,
    llm=integrated_llm,
)
