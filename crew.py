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
    facility_cooling_task,
    facility_energy_task,
    idle_facility_task,
    integrated_task,
    integrated_compute_task,
    integrated_cooling_task,
    integrated_energy_task,
    power_task,
    scheduler_task,
)


ARCHITECTURES = ("four_agent", "three_agent", "two_agent", "single_agent")


def get_data_center_crew(
    architecture="four_agent",
    skip_compute=False,
    skip_cooling=False,
    skip_battery=False,
):
    """Build the selected CrewAI architecture."""
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}; "
            f"choose one of {ARCHITECTURES}."
        )

    agents = []
    tasks = []
    if architecture == "four_agent":
        if not skip_compute:
            agents.extend([scheduler_agent, power_governor_agent])
            tasks.extend([scheduler_task, power_task])
        if not skip_cooling:
            agents.append(cooling_agent)
            tasks.append(cooling_task_four)
        if not skip_battery:
            agents.append(energy_agent)
            tasks.append(energy_task_four)
    elif architecture == "three_agent":
        if not skip_compute:
            agents.append(compute_agent)
            tasks.append(compute_task)
        if not skip_cooling:
            agents.append(cooling_agent)
            tasks.append(cooling_task_three)
        if not skip_battery:
            agents.append(energy_agent)
            tasks.append(energy_task_three)
    elif architecture == "two_agent":
        if not skip_compute:
            agents.append(compute_agent)
            tasks.append(compute_task)
        if not skip_cooling and not skip_battery:
            agents.append(facility_energy_agent)
            tasks.append(facility_energy_task)
        elif not skip_cooling:
            agents.append(facility_energy_agent)
            tasks.append(facility_cooling_task)
        elif not skip_battery:
            agents.append(facility_energy_agent)
            tasks.append(facility_battery_task)
    elif not (skip_compute or skip_cooling or skip_battery):
        agents = [integrated_ems_agent]
        tasks = [integrated_task]
    elif skip_compute and not skip_cooling and not skip_battery:
        agents = [integrated_ems_agent]
        tasks = [idle_facility_task]
    else:
        agents = [integrated_ems_agent]
        if not skip_compute:
            tasks.append(integrated_compute_task)
        if not skip_cooling:
            tasks.append(integrated_cooling_task)
        if not skip_battery:
            tasks.append(integrated_energy_task)

    if not tasks:
        raise ValueError("All AI decisions were skipped; no crew is required.")

    return Crew(
        # CrewAI's constructor is typed as list[BaseAgent], while these
        # concrete objects are list[Agent]. Runtime behavior is valid, but
        # list invariance prevents Pylance from accepting the narrower list.
        agents=cast(Any, agents),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
