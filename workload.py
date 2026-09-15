from state import state


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
    state.batch_backlog_cpu = sum(job["cpu"] for job in state.batch_backlog)
    state.batch_served_cpu = state.batch_backlog_cpu
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
    state.batch_served_cpu = max(due_cpu, state.batch_backlog_cpu * fraction)
    state.pending_workload_cpu = (
        state.interactive_workload_cpu + state.batch_served_cpu
    )
    return state.pending_workload_cpu


def advance_batch_queue():
    """Remove served work FIFO, age the remainder, and count deadline misses."""
    remaining_service = state.batch_served_cpu
    for job in state.batch_backlog:
        served = min(job["cpu"], remaining_service)
        job["cpu"] -= served
        remaining_service -= served
    state.batch_backlog = [j for j in state.batch_backlog if j["cpu"] > 1e-9]
    missed = 0.0
    for job in state.batch_backlog:
        job["hours_left"] -= 1
        if job["hours_left"] <= 0:
            missed += job["cpu"]
    state.batch_backlog = [j for j in state.batch_backlog if j["hours_left"] > 0]
    state.batch_deadline_missed_cpu = missed
    state.batch_backlog_cpu = sum(job["cpu"] for job in state.batch_backlog)
    return missed
