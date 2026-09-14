from typing import Any, cast

from crewai import Crew, Process

from agents import (
    compute_agent,
    cooling_agent,
    energy_agent,
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
    integrated_task,
    power_task,
    scheduler_task,
)


ARCHITECTURES = ("four_agent", "three_agent", "single_agent")


def get_data_center_crew(architecture="four_agent"):
    """Build the selected CrewAI architecture."""
    if architecture == "four_agent":
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
    elif architecture == "single_agent":
        agents = [integrated_ems_agent]
        tasks = [integrated_task]
    else:
        raise ValueError(
            f"Unknown architecture {architecture!r}; "
            f"choose one of {ARCHITECTURES}."
        )

    return Crew(
        # CrewAI's constructor is typed as list[BaseAgent], while these
        # concrete objects are list[Agent]. Runtime behavior is valid, but
        # list invariance prevents Pylance from accepting the narrower list.
        agents=cast(Any, agents),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

