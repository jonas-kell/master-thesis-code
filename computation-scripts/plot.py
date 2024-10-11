from typing import Dict, Union, Any, Tuple, List, cast, Literal
import numpy as np
import os
import json


def plot_measurements(
    times: List[float],
    values: List[List[float]],
    observable_labels: List[str],
    title: str,
    x_label: str,
    params: Tuple[
        float,  # U
        float,  # E
        float,  # J
    ],
    time_unit_type: Literal["unscaled", "one_over_U", "one_over_J", "one_over_E"] = (
        "one_over_J"
    ),
):
    import matplotlib.pyplot as plt

    if time_unit_type == "one_over_U":  # type: ignore - switch is hard-coded.
        time_scaler = float(np.abs(params[0]))
        label_explanation = "1/U"
    elif time_unit_type == "one_over_E":  # type: ignore - switch is hard-coded.
        time_scaler = float(np.abs(params[1]))
        label_explanation = "1/E"
    elif time_unit_type == "one_over_J":  # type: ignore - switch is hard-coded.
        if float(np.abs(params[2])) < 1e-5:
            # backup chose U if interaction with J is 0
            time_scaler = float(np.abs(params[0]))
            label_explanation = "1/U"
        else:
            time_scaler = float(np.abs(params[2]))
            label_explanation = "1/J"
    else:
        time_scaler = 1.0
        label_explanation = "unscaled units"

    times_scaled = np.array(times) * time_scaler

    inported_data = np.array(values).T  # Transpose to extract form we want to plot

    num_observables = len(observable_labels)
    num_rows = max(
        1, int(np.ceil(num_observables / np.sqrt(num_observables)))
    )  # Divide by root to get approximate square arrangement
    num_cols = int(np.ceil(num_observables / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 6))  # type: ignore -> matplotlib typing is non-existent
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])  # type: ignore -> matplotlib typing is non-existent
    else:
        if num_rows == 1 or num_cols == 1:
            axes = np.array([axes]).T  # type: ignore -> matplotlib typing is non-existent

    for i, obs in enumerate(observable_labels):
        row = i // num_cols
        col = i % num_cols

        # Plot the results
        axes[row, col].plot(times_scaled, inported_data[i], color="red")  # type: ignore -> matplotlib typing is non-existent
        axes[row, col].set_xlabel(f"{x_label} in {label_explanation}")  # type: ignore -> matplotlib typing is non-existent
        axes[row, col].set_ylabel(obs)  # type: ignore -> matplotlib typing is non-existent

    # Remove any unused subplots
    for i in range(num_observables, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    plt.suptitle(title)  # type: ignore -> matplotlib typing is non-existent
    plt.tight_layout()  # type: ignore -> matplotlib typing is non-existent
    plt.show()  # type: ignore -> matplotlib typing is non-existent


if __name__ == "__main__":
    file_name = "2024-XX-XX__XX,XX,XX.json"
    current_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./../run-outputs/" + file_name,
    )

    print(f"plotting measurements from file: {current_file_path}")

    loaded_data: Dict[str, Union[float, str, Dict[Any, Any], List[Any]]] = {}
    with open(current_file_path, mode="r") as file:
        loaded_data = json.load(file)

    loaded_title: str = loaded_data["plot_title"]  # type: ignore
    plot_x_label: str = loaded_data["plot_x_label"]  # type: ignore

    U: float = float(loaded_data["hamiltonian"]["U"])  # type: ignore
    E: float = float(loaded_data["hamiltonian"]["E"])  # type: ignore
    J: float = float(loaded_data["hamiltonian"]["J"])  # type: ignore

    times_list: List[float] = []
    values_list: List[List[float]] = []
    for tmp in loaded_data["measurements"]:  # type: ignore
        time = cast(float, tmp["time"])
        value = cast(List[float], tmp["data"])

        times_list.append(time)
        values_list.append(value)

    observable_labels_list: List[str] = [tmp["label"] for tmp in loaded_data["observables"]]  # type: ignore

    plot_measurements(
        observable_labels=observable_labels_list,
        times=times_list,
        values=values_list,
        title=loaded_title,
        x_label=plot_x_label,
        params=(U, E, J),
    )
