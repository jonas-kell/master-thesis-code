from typing import List, Any, Dict, Union, Tuple
import numpy as np
from commonsettings import (
    get_full_file_path,
    scaler_switch,
    get_newest_file_name,
    get_filenames_containing,
)
import json


def scale_and_prepare_data(
    observable_type, J, U, data: np.ndarray
) -> Tuple[np.ndarray, str]:
    rescale_orders = np.abs(J) > scaler_switch

    second_order_error_factor = 1
    if rescale_orders:
        second_order_error_factor = (U / J) ** 2

    center_var = 0
    scale_factor = 1
    label_addendum = ""

    if observable_type == "SpinCurrent":
        scale_factor = 1 / J
        center_var = 0
        label_addendum = "in J"
    elif observable_type == "DoubleOccupationAtSite":
        if rescale_orders:
            scale_factor = second_order_error_factor
            label_addendum = "in J²/U² around 0.25"
        center_var = 0.25
    elif observable_type == "OccupationAtSite":
        if rescale_orders:
            scale_factor = second_order_error_factor
            label_addendum = "in J²/U² around 0.5"
        center_var = 0.5
    elif observable_type == "Purity":
        pass
    elif observable_type == "Concurrence":
        pass
    elif observable_type == "ConcurrenceAsymm":
        pass
    elif observable_type == "PauliMeasurement":
        pass
    elif observable_type == "VCNFactor":
        pass
    elif observable_type == "BaseEnergyFactor":
        pass
    elif observable_type == "Energy":
        pass
    elif observable_type == "EnergyVariance":
        pass
    elif observable_type == "NormalizationComparison":
        pass
    elif observable_type == "LocalKinEnergyEquivalent":
        pass
    else:
        raise Exception(f"Unknown Preparation Measurement {observable_type}")

    return ((data - center_var) * scale_factor, " " + label_addendum)


def plot_experiment_comparison(
    filenames: List[str],
    replacement_names=None,
    difference_tuples: List[Tuple[int, int]] = [],
):
    # ! only require this import when we are plotting
    import matplotlib.pyplot as plt

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

        difference_cache = [(None, None) for _ in difference_tuples]
        for j, filename in enumerate(filenames):
            times_to_plot = []
            values_to_plot = []

            for measurement in data_array[j]:
                times_to_plot.append(measurement["time"])
                values_to_plot.append(measurement["data"][i])

            scaled_data, label_addendum = scale_and_prepare_data(
                observable["type"], J, U, np.array(values_to_plot)
            )
            scaled_times = np.array(times_to_plot) * scaler_factor

            for chache_index, (
                difference_tuple_index_0,
                difference_tuple_index_1,
            ) in enumerate(difference_tuples):
                mod = list(difference_cache[chache_index])
                if j == difference_tuple_index_0:
                    mod[0] = scaled_times, scaled_data
                if j == difference_tuple_index_1:
                    mod[1] = scaled_times, scaled_data
                difference_cache[chache_index] = tuple(mod)

            plt.plot(
                scaled_times,
                scaled_data,
                label=replacement_names[j],
            )

        for difference_index, ((times_0, data_0), (times_1, data_1)) in enumerate(
            difference_cache
        ):
            _ = times_1
            plt.plot(
                times_0,  # assume times_1 here to be the same
                np.abs(data_0 - data_1),
                label=f"difference {difference_index}",
            )

        # Adding labels and title
        plt.xlabel("Time in 1/" + scaler_factor_label)
        plt.ylabel(label + label_addendum)
        plt.legend()

        # Show the plot
        plt.show()


if __name__ == "__main__":

    # test = "2025-01-25__13,51,22"
    # test = "2025-01-25__13,52,28"
    # test = "2025-01-25__13,54,03"
    # test = "2025-01-25__13,54,20"
    test = "2025-01-25__13,54,34"
    # test = "2025-01-25__13,54,57"
    # test = "2025-01-25__13,55,46"
    # test = "2025-01-25__13,57,26"
    # test = "2025-01-25__14,01,04"
    # test = "2025-01-25__14,01,18"
    # test = "2025-01-25__14,04,38"
    # test = "2025-01-25__14,05,15"
    # test = "2025-01-25__14,19,17"
    # test = "2025-01-25__14,19,52"
    # test = "2025-01-25__14,20,57"
    # test = "2025-01-25__14,21,02"
    # test = "2025-01-25__14,22,48"
    # test = "2025-01-25__14,36,01"
    # test = "2025-01-25__14,36,06"
    # test = "2025-01-25__14,37,41"
    # test = "2025-01-25__14,37,46"
    # test = "2025-01-25__14,42,18"
    # test = "2025-01-25__15,14,39"
    # test = "2025-01-25__15,17,02"
    # test = "2025-01-25__15,17,39"
    # test = "2025-01-25__15,18,25"
    # test = "2025-01-25__15,19,27"
    # test = "2025-01-25__15,19,47"
    # test = "2025-01-25__15,20,45"
    # test = "2025-01-25__15,21,07"
    # test = "2025-01-25__15,21,44"
    # test = "2025-01-25__15,22,24"
    # test = "2025-01-25__15,27,14"
    # test = "2025-01-25__15,27,45"
    # test = "2025-01-25__15,43,56"
    # test = "2025-01-25__15,45,23"
    # test = "2025-01-25__15,47,21"
    # test = "2025-01-25__15,52,02"
    # test = "2025-01-25__15,52,51"
    # test = "2025-01-25__16,04,15"
    # test = "2025-01-25__16,18,07"
    # test = "2025-01-25__16,20,14"
    # test = "2025-01-25__16,21,46"
    # test = "2025-01-25__16,22,09"
    # test = "2025-01-25__16,29,06"
    # test = "2025-01-25__18,06,54"
    # test = "2025-01-25__18,11,06"
    # test = "2025-01-25__18,12,11"
    # test = "2025-01-25__18,14,36"
    # test = "2025-01-25__18,15,37"
    # test = "2025-01-25__18,16,17"
    # test = "2025-01-25__18,17,03"
    # test = "2025-01-25__18,17,47"
    # test = "2025-01-25__18,18,40"
    # test = "2025-01-25__18,19,47"
    # test = "2025-01-25__18,20,29"
    # test = "2025-01-25__18,22,30"
    # test = "2025-01-25__18,27,13"
    # test = "2025-01-25__18,27,33"
    # test = "2025-01-25__18,50,43"
    # test = "2025-01-25__18,51,54"
    # test = "2025-01-25__18,52,44"
    # test = "2025-01-25__18,53,13"
    # test = "2025-01-25__19,04,28"
    # test = "2025-01-25__19,05,23"
    # test = "2025-01-25__19,06,05"
    # test = "2025-01-25__19,12,13"
    # test = "2025-01-25__19,13,33"
    # test = "2025-01-25__19,13,57"
    # test = "2025-01-25__19,15,04"
    # test = "2025-01-25__19,15,24"
    # test = "2025-01-25__19,15,49"
    # test = "2025-01-25__19,18,48"
    # test = "2025-01-25__19,28,59"
    # test = "2025-01-25__19,29,17"
    # test = "2025-01-25__19,29,38"

    list_of_filenames = get_filenames_containing(test)

    # diagonalization = "diagonalization_measurements_2024-11-26__11,44,27"
    plot_experiment_comparison(
        list_of_filenames,
    )
