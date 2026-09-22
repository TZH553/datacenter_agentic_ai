from typing import Any, cast

from crewai import Crew, Process

from agents import (
    compute_agent,
    cooling_agent,
    energy_agent,
    facility_energy_agent,
    integrated_ems_agent,
    power_governor_agent,
    scheduler_agent,
)
from tasks import (
    compute_task,
    cooling_task_four,
    cooling_task_three,
    energy_task_four,
    energy_task_three,
    facility_battery_task,
    facility_energy_task,
    idle_facility_task,
    integrated_task,
    integrated_compute_task,
    integrated_energy_task,
    power_task,
    scheduler_task,
)


ARCHITECTURES = ("four_agent", "three_agent", "two_agent", "single_agent")


def get_data_center_crew(
    architecture="four_agent",
    skip_compute=False,
    skip_cooling=False,
):
    """Build the selected CrewAI architecture."""
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}; "
            f"choose one of {ARCHITECTURES}."
        )

    if skip_compute and skip_cooling and architecture == "four_agent":
        agents = [energy_agent]
        tasks = [energy_task_four]
    elif skip_compute and skip_cooling and architecture == "three_agent":
        agents = [energy_agent]
        tasks = [energy_task_three]
    elif skip_compute and skip_cooling and architecture == "two_agent":
        agents = [facility_energy_agent]
        tasks = [facility_battery_task]
    elif skip_compute and skip_cooling and architecture == "single_agent":
        agents = [integrated_ems_agent]
        tasks = [integrated_energy_task]
    elif skip_cooling and architecture == "four_agent":
        agents = [scheduler_agent, power_governor_agent, energy_agent]
        tasks = [scheduler_task, power_task, energy_task_four]
    elif skip_cooling and architecture == "three_agent":
        agents = [compute_agent, energy_agent]
        tasks = [compute_task, energy_task_three]
    elif skip_cooling and architecture == "two_agent":
        agents = [compute_agent, facility_energy_agent]
        tasks = [compute_task, facility_battery_task]
    elif skip_cooling and architecture == "single_agent":
        agents = [integrated_ems_agent]
        tasks = [integrated_compute_task, integrated_energy_task]
    elif skip_compute and architecture == "four_agent":
        agents = [cooling_agent, energy_agent]
        tasks = [cooling_task_four, energy_task_four]
    elif skip_compute and architecture == "three_agent":
        agents = [cooling_agent, energy_agent]
        tasks = [cooling_task_three, energy_task_three]
    elif skip_compute and architecture == "two_agent":
        agents = [facility_energy_agent]
        tasks = [facility_energy_task]
    elif skip_compute and architecture == "single_agent":
        agents = [integrated_ems_agent]
        tasks = [idle_facility_task]
    elif architecture == "four_agent":
        agents = [
            scheduler_agent,
            power_governor_agent,
            cooling_agent,
            energy_agent,
        ]
        tasks = [
            scheduler_task,
            power_task,
            cooling_task_four,
            energy_task_four,
        ]
    elif architecture == "three_agent":
        agents = [compute_agent, cooling_agent, energy_agent]
        tasks = [compute_task, cooling_task_three, energy_task_three]
    elif architecture == "two_agent":
        # Agent 1 owns scheduling + consolidation; Agent 2 coordinates the
        # thermal and electrical subsystems.
        agents = [compute_agent, facility_energy_agent]
        tasks = [compute_task, facility_energy_task]
    elif architecture == "single_agent":
        agents = [integrated_ems_agent]
        tasks = [integrated_task]

    return Crew(
        # CrewAI's constructor is typed as list[BaseAgent], while these
        # concrete objects are list[Agent]. Runtime behavior is valid, but
        # list invariance prevents Pylance from accepting the narrower list.
        agents=cast(Any, agents),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
