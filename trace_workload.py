from pathlib import Path

import pandas as pd

from config import Config


REQUIRED_COLUMNS = {
    "Job_ID",
    "Submit_Time",
    "Used_CPUs",
    "Execution_Time(Seconds)",
    "Job_Type",
    "Priority_Level",
}
PRIORITY_INDEX = {"high": 0, "medium": 1, "low": 2}


def _total_cpu_capacity(config: Config) -> float:
    if config.clusters is not None:
        return sum(cluster.cpu_capacity for cluster in config.clusters)
    cluster_capacity = (
        config.cpu_capacity
        if config.cpu_capacity is not None
        else (
            config.servers_per_cluster
            * config.processing_capacity_per_server
        )
    )
    return config.n_clusters * cluster_capacity


def _deadline_hours(job_type: str, priority: str, config: Config) -> int:
    priority_key = str(priority).strip().lower()
    if priority_key not in PRIORITY_INDEX:
        raise ValueError(f"Unsupported priority level: {priority!r}")
    priority_index = PRIORITY_INDEX[priority_key]
    type_key = str(job_type).strip().lower()
    if type_key == "gpu":
        deadlines = config.trace_gpu_deadlines_h
    elif type_key == "mpi":
        deadlines = config.trace_mpi_deadlines_h
    else:
        deadlines = config.trace_batch_deadlines_h
    deadline = int(deadlines[priority_index])
    if deadline <= 0:
        raise ValueError("Trace deadlines must be positive hours.")
    return deadline


def load_cloud_workload_trace(
    csv_path: str | Path,
    hours: int,
    config: Config,
) -> tuple[list[float], list[dict]]:
    """Convert job records into one-hour workload-arrival bins.

    Work is represented as CPU-core-hours:
    Used_CPUs * Execution_Time(Seconds) / 3600.
    Interactive jobs must run in their arrival bin. Batch, GPU and MPI jobs
    enter the flexible queue with type- and priority-specific deadlines.
    """
    if hours <= 0:
        raise ValueError("hours must be positive.")
    if abs(config.timestep_h - 1.0) > 1e-9:
        raise ValueError(
            "Cloud trace loading currently requires timestep_h = 1.0 hour."
        )
    if config.trace_scale_factor <= 0:
        raise ValueError("trace_scale_factor must be positive.")

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cloud workload trace not found: {path.resolve()}"
        )

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Cloud workload trace is missing columns: {sorted(missing)}"
        )

    frame["Submit_Time"] = pd.to_datetime(
        frame["Submit_Time"], errors="raise"
    )
    numeric_columns = ["Used_CPUs", "Execution_Time(Seconds)"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame[numeric_columns] < 0).any().any():
        raise ValueError("CPU use and execution time cannot be negative.")

    frame["cpu_work"] = (
        frame["Used_CPUs"]
        * frame["Execution_Time(Seconds)"]
        / 3600.0
        * config.trace_scale_factor
    )

    if config.trace_window_start is None:
        window_start = frame["Submit_Time"].min().floor("h")
    else:
        window_start = pd.Timestamp(config.trace_window_start).floor("h")
    window_end = window_start + pd.Timedelta(hours=hours)
    selected = frame[
        (frame["Submit_Time"] >= window_start)
        & (frame["Submit_Time"] < window_end)
    ].copy()
    selected["bin"] = selected["Submit_Time"].dt.floor("h")

    total_capacity = _total_cpu_capacity(config)
    workload_profile: list[float] = []
    arrivals: list[dict] = []

    for offset in range(hours):
        bin_start = window_start + pd.Timedelta(hours=offset)
        jobs = selected[selected["bin"] == bin_start]
        interactive_cpu = 0.0
        batch_jobs = []

        for row in jobs.itertuples(index=False):
            cpu_work = float(row.cpu_work)
            job_type = str(row.Job_Type).strip()
            priority = str(row.Priority_Level).strip().lower()
            if job_type.lower() == "interactive":
                interactive_cpu += cpu_work
                continue
            batch_jobs.append(
                {
                    "job_id": str(row.Job_ID),
                    "cpu": cpu_work,
                    "hours_left": _deadline_hours(
                        job_type, priority, config
                    ),
                    "job_type": job_type,
                    "priority": priority,
                }
            )

        batch_cpu = sum(job["cpu"] for job in batch_jobs)
        arrival_cpu = interactive_cpu + batch_cpu
        workload_profile.append(
            min(1.0, arrival_cpu / max(total_capacity, 1e-9))
        )
        arrivals.append(
            {
                "bin_start": bin_start.isoformat(),
                "interactive_cpu": interactive_cpu,
                "batch_jobs": batch_jobs,
                "arrival_cpu": arrival_cpu,
            }
        )

    return workload_profile, arrivals
