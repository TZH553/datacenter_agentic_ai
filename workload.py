from state import state


def add_workload(workload_fraction):
    """Convert normalized demand into explicit CPU workload units.

    For example, a value of 0.70 with 1,000 installed CPU units produces
    700 pending CPU units. Cluster capacities may be different.
    """
    fraction = max(0.0, min(1.0, float(workload_fraction)))
    state.pending_workload_fraction = fraction
    state.pending_workload_cpu = fraction * state.total_cpu_capacity
    return state.pending_workload_cpu
