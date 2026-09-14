from crewai import Crew, Process

from agents import (
    scheduler_agent,
    power_governor_agent,
    cooling_agent,
    energy_agent
)

from tasks import (
    scheduler_task,
    power_task,
    cooling_task,
    energy_task
)


data_center_crew = Crew(

    agents=[
        scheduler_agent,
        power_governor_agent,
        cooling_agent,
        energy_agent
    ],

    tasks=[
        scheduler_task,
        power_task,
        cooling_task,
        energy_task
    ],

    process=Process.sequential,

    verbose=True
)