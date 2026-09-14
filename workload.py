from state import state


def add_workload(workload):
    """
    Add incoming workload to the pending workload queue.

    workload:
        normalized value between 0 and 1

    Example:
        0.70 = 70% of total data-centre capacity
    """

    workload = max(
        0.0,
        min(1.0, float(workload))
    )

    state.pending_workload = workload

    return state.pending_workload