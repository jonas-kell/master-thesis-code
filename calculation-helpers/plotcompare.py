from typing import List, Any, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
from commonsettings import get_full_file_path, scaler_switch
import json


def plot_experiment_comparison(
    filenames: List[str],
    replacement_names=None,
):
    if replacement_names is None or len(replacement_names) != len(filenames):
        replacement_names = filenames

    loaded_data_array: List[Dict[str, Union[float, str, Dict[Any, Any], List[Any]]]] = (
        []
    )
    for filename in filenames:
        with open(get_full_file_path(filename), mode="r") as file:
            loaded_data_array.append(json.load(file))

    J = loaded_data_array[0]["hamiltonian"]["J"]
    U = loaded_data_array[0]["hamiltonian"]["U"]

    data_array = []
    for i, filename in enumerate(filenames):
        print("reading: ", filename)
        data_array.append(loaded_data_array[i]["measurements"])

        Jcomp = loaded_data_array[i]["hamiltonian"]["J"]
        Ucomp = loaded_data_array[i]["hamiltonian"]["U"]

        if Jcomp != J or Ucomp != U:
            print("Not comparable Files?")

    if np.abs(J) < scaler_switch:
        # if J interaction "deactivated", scale with U
        scaler_factor = U
        scaler_factor_label = "U"
    else:
        scaler_factor = J
        scaler_factor_label = "J"

    observables = loaded_data_array[0]["observables"]

    for i, observable in enumerate(observables):
        label = observable["label"]

        for j, filename in enumerate(filenames):
            times_to_plot = []
            values_to_plot = []

            for measurement in data_array[j]:
                times_to_plot.append(measurement["time"])
                values_to_plot.append(measurement["data"][i])

            plt.plot(
                np.array(times_to_plot) * scaler_factor,
                values_to_plot,
                label=replacement_names[j],
            )

        # Adding labels and title
        plt.xlabel("Time in 1/" + scaler_factor_label)
        plt.ylabel(label)
        plt.legend()

        # Show the plot
        plt.show()


if __name__ == "__main__":

    # was the flip correction justified
    time_string = "2024-10-23__23,26,46"  # no flip correction
    time_string = "2024-10-23__23,30,23"  # flip correction

    # things drastically change from 3 to 4 elements in the chain
    time_string = "2024-10-23__23,49,06"  # 3
    time_string = "2024-10-23__23,50,51"  # 4

    time_string = (
        "2024-10-24__23,12,46"  # un-centered indicees, not locationally invertable
    )
    time_string = "2024-10-24__23,06,39"  # centered indicees, locationally invertable
    time_string = "2024-10-25__09,07,00"  # diagonalization indicees also centered

    # filename_a = "perturbation_measurements_" + time_string
    # filename_b = "diagonalization_measurements_" + time_string

    # plot_experiment_comparison([filename_a, filename_b])

    diagonal = "diagonalization_measurements_2024-11-01__16,14,33"
    first_order = "perturbation_measurements_2024-11-01__16,09,01"
    second_order = "perturbation_measurements_2024-11-01__16,14,33"

    plot_experiment_comparison(
        [diagonal, first_order, second_order],
        ["diagonal", "pert. first", "pert. second"],
    )
