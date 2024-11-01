from typing import List, Any, Dict, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
from commonsettings import get_full_file_path, scaler_switch
import json


def scale_and_prepare_data(
    observable_type, J, U, data: np.ndarray
) -> Tuple[np.ndarray, str]:
    second_order_error_factor = (U / J) ** 2

    if np.abs(J) < scaler_switch:
        raise Exception("Can not rescale by this, as J-deactivated")

    center_var = 0
    scale_factor = 1
    label_addendum = ""

    if observable_type == "SpinCurrent":
        scale_factor = 1 / J
        center_var = 0
        label_addendum = "in J"
    elif observable_type == "DoubleOccupationAtSite":
        scale_factor = second_order_error_factor
        center_var = 0.25
        label_addendum = "in J²/U² around 0.25"
    elif observable_type == "OccupationAtSite":
        scale_factor = second_order_error_factor
        center_var = 0.5
        label_addendum = "in J²/U² around 0.5"
    elif observable_type == "Purity":
        pass
    elif observable_type == "Concurrence":
        pass
    elif observable_type == "ConcurrenceAsymm":
        pass
    elif observable_type == "PauliMeasurement":
        pass
    else:
        raise Exception(f"Unknown Preparation Measurement {observable_type}")

    return ((data - center_var) * scale_factor, " " + label_addendum)


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
        print(loaded_data_array[i]["hamiltonian"])

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

            scaled_data, label_addendum = scale_and_prepare_data(
                observable["type"], J, U, np.array(values_to_plot)
            )

            plt.plot(
                np.array(times_to_plot) * scaler_factor,
                (scaled_data),
                label=replacement_names[j],
            )

        # Adding labels and title
        plt.xlabel("Time in 1/" + scaler_factor_label)
        plt.ylabel(label + label_addendum)
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

    # spinToZBasisTransformation
    exact = "diagonalization_measurements_2024-11-01__21,29,39"
    current = "perturbation_measurements_2024-11-01__21,29,39"
    flipped = "perturbation_measurements_2024-11-01__21,25,14"
    plot_experiment_comparison(
        [
            exact,
            current,
            flipped,
        ],
        [
            "exact",
            "current",
            "flipped",
        ],
    )

    # first second order comparison
    # diagonal = "diagonalization_measurements_2024-11-01__16,14,33"
    # first_order = "perturbation_measurements_2024-11-01__16,09,01"
    # second_order = "perturbation_measurements_2024-11-01__16,14,33"
    # plot_experiment_comparison(
    #     [diagonal, first_order, second_order],
    #     ["diagonal", "pert. first", "pert. second"],
    # )
