from state import state


def apply_action(action):

    for host_name, active in action["host_power"].items():

        if host_name in state.hosts:
            state.hosts[host_name]["active"] = active


    for host_name, utilisation in action["host_utilisation"].items():

        if host_name in state.hosts:

            if state.hosts[host_name]["active"]:
                state.hosts[host_name]["utilisation"] = utilisation
            else:
                state.hosts[host_name]["utilisation"] = 0.0


    state.cooling_factor = action["cooling_factor"]

    state.battery_command_kw = action["battery_command_kw"]