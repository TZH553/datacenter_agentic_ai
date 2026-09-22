from state import state


def _plan_batch_service(requested_cpu):
    """Build an EDF service plan constrained by hourly GPU capacity."""
    ordered_jobs = sorted(
        state.batch_backlog,
        key=lambda queued_job: queued_job["hours_left"],
    )

    # First calculate the GPU demand implied by the requested CPU service.
    remaining_requested_cpu = max(0.0, float(requested_cpu))
    requested_gpu_work = state.interactive_gpu_work
    for job in ordered_jobs:
        desired_cpu = min(job["cpu"], remaining_requested_cpu)
        if job["cpu"] > 1e-9:
            requested_gpu_work += (
                job.get("gpu_work", 0.0) * desired_cpu / job["cpu"]
            )
        remaining_requested_cpu -= desired_cpu
        if remaining_requested_cpu <= 1e-9:
            break

    # Then produce the feasible plan. CPU-only work can still run when all
    # GPUs are occupied, so later CPU-only jobs are not unnecessarily blocked.
    remaining_cpu = max(0.0, float(requested_cpu))
    remaining_gpu_work = max(
        0.0,
        state.total_gpu_capacity * state.timestep_h
        - state.interactive_gpu_work,
    )
    selected_gpu_work = min(
        state.interactive_gpu_work,
        state.total_gpu_capacity * state.timestep_h,
    )
    selected_cpu = 0.0
    for job in ordered_jobs:
        desired_cpu = min(job["cpu"], remaining_cpu)
        gpu_per_cpu = (
            job.get("gpu_work", 0.0) / job["cpu"]
            if job["cpu"] > 1e-9
            else 0.0
        )
        if gpu_per_cpu > 0.0:
            served_cpu = min(
                desired_cpu,
                remaining_gpu_work / gpu_per_cpu,
            )
        else:
            served_cpu = desired_cpu
        job["_selected_cpu"] = served_cpu
        selected_cpu += served_cpu
        remaining_cpu -= served_cpu
        served_gpu_work = served_cpu * gpu_per_cpu
        selected_gpu_work += served_gpu_work
        remaining_gpu_work -= served_gpu_work

    state.batch_served_cpu = selected_cpu
    state.scheduled_gpu_work = selected_gpu_work
    state.requested_gpu_demand = requested_gpu_work / state.timestep_h
    state.gpu_demand = selected_gpu_work / state.timestep_h


def add_workload(workload_fraction):
    """Convert normalized demand into explicit CPU workload units.

    For example, a value of 0.70 with 1,000 installed CPU units produces
    700 pending CPU units. Cluster capacities may be different.
    """
    fraction = max(0.0, min(1.0, float(workload_fraction)))
    state.pending_workload_fraction = fraction
    arrival_cpu = fraction * state.total_cpu_capacity
    state.interactive_workload_cpu = (
        arrival_cpu * state.interactive_workload_fraction
    )
    state.batch_arrival_cpu = arrival_cpu - state.interactive_workload_cpu
    if state.batch_arrival_cpu > 1e-9:
        state.batch_backlog.append(
            {"cpu": state.batch_arrival_cpu, "hours_left": state.batch_deadline_h}
        )
    state.interactive_gpu_work = 0.0
    state.batch_arrival_gpu_work = 0.0
    state.batch_backlog_gpu_work = 0.0
    state.batch_backlog_cpu = sum(job["cpu"] for job in state.batch_backlog)
    _plan_batch_service(state.batch_backlog_cpu)
    state.pending_workload_cpu = (
        state.interactive_workload_cpu + state.batch_served_cpu
    )
    return state.pending_workload_cpu


def add_trace_workload(hour_arrivals):
    """Add one hourly bin of explicit trace jobs to the scheduler state."""
    interactive_cpu = max(
        0.0, float(hour_arrivals.get("interactive_cpu", 0.0))
    )
    batch_jobs = hour_arrivals.get("batch_jobs", [])

    state.interactive_workload_cpu = interactive_cpu
    state.interactive_gpu_work = max(
        0.0, float(hour_arrivals.get("interactive_gpu_work", 0.0))
    )
    state.batch_arrival_cpu = 0.0
    state.batch_arrival_gpu_work = 0.0
    for source_job in batch_jobs:
        cpu_work = max(0.0, float(source_job["cpu"]))
        if cpu_work <= 1e-9:
            continue
        hours_left = int(source_job["hours_left"])
        if hours_left <= 0:
            raise ValueError("Trace job deadline must be positive.")
        job = dict(source_job)
        job["cpu"] = cpu_work
        job["hours_left"] = hours_left
        state.batch_backlog.append(job)
        state.batch_arrival_cpu += cpu_work
        state.batch_arrival_gpu_work += max(
            0.0, float(job.get("gpu_work", 0.0))
        )

    arrival_cpu = (
        state.interactive_workload_cpu + state.batch_arrival_cpu
    )
    state.pending_workload_fraction = min(
        1.0, arrival_cpu / max(state.total_cpu_capacity, 1e-9)
    )
    state.batch_backlog_cpu = sum(
        job["cpu"] for job in state.batch_backlog
    )
    state.batch_backlog_gpu_work = sum(
        job.get("gpu_work", 0.0) for job in state.batch_backlog
    )
    _plan_batch_service(state.batch_backlog_cpu)
    state.pending_workload_cpu = (
        state.interactive_workload_cpu + state.batch_served_cpu
    )
    return state.pending_workload_cpu


def select_batch_service(service_fraction):
    """Select flexible batch demand; jobs due now are always forced to run."""
    fraction = max(0.0, min(1.0, float(service_fraction)))
    due_cpu = sum(
        job["cpu"] for job in state.batch_backlog if job["hours_left"] <= 1
    )
    requested_cpu = max(due_cpu, state.batch_backlog_cpu * fraction)
    _plan_batch_service(requested_cpu)
    state.pending_workload_cpu = (
        state.interactive_workload_cpu + state.batch_served_cpu
    )
    return state.pending_workload_cpu


def advance_batch_queue():
    """Serve earliest-deadline jobs, then age and expire the remainder."""
    service_order = sorted(
        state.batch_backlog,
        key=lambda job: job["hours_left"],
    )
    for job in service_order:
        served = min(job["cpu"], job.pop("_selected_cpu", 0.0))
        cpu_before = job["cpu"]
        if cpu_before > 1e-9:
            job["gpu_work"] = job.get("gpu_work", 0.0) * (
                1.0 - served / cpu_before
            )
        job["cpu"] -= served
    state.batch_backlog = [j for j in state.batch_backlog if j["cpu"] > 1e-9]
    missed = 0.0
    for job in state.batch_backlog:
        job["hours_left"] -= 1
        if job["hours_left"] <= 0:
            missed += job["cpu"]
    state.batch_backlog = [j for j in state.batch_backlog if j["hours_left"] > 0]
    state.batch_deadline_missed_cpu = missed
    state.batch_backlog_cpu = sum(job["cpu"] for job in state.batch_backlog)
    state.batch_backlog_gpu_work = sum(
        job.get("gpu_work", 0.0) for job in state.batch_backlog
    )
    return missed
