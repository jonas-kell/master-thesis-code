from typing import List, Any, Dict, Union, Tuple
import numpy as np
from commonsettings import get_full_file_path, scaler_switch, get_newest_file_name
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

    # # was the flip correction justified
    # time_string = "2024-10-23__23,26,46"  # no flip correction
    # time_string = "2024-10-23__23,30,23"  # flip correction

    # # things drastically change from 3 to 4 elements in the chain
    # time_string = "2024-10-23__23,49,06"  # 3
    # time_string = "2024-10-23__23,50,51"  # 4

    # time_string = (
    #     "2024-10-24__23,12,46"  # un-centered indicees, not locationally invertable
    # )
    # time_string = "2024-10-24__23,06,39"  # centered indicees, locationally invertable
    # time_string = "2024-10-25__09,07,00"  # diagonalization indicees also centered

    # filename_a = "perturbation_measurements_" + time_string
    # filename_b = "diagonalization_measurements_" + time_string

    # spinToZBasisTransformation
    # exact = "diagonalization_measurements_2024-11-01__21,29,39"
    # current = "perturbation_measurements_2024-11-01__21,29,39"
    # flipped = "perturbation_measurements_2024-11-01__21,25,14"
    # plot_experiment_comparison(
    #     [
    #         exact,
    #         current,
    #         flipped,
    #     ],
    #     [
    #         "exact",
    #         "current",
    #         "flipped",
    #     ],
    # )

    # skip non-knowing terms test
    # diagonal = "diagonalization_measurements_2024-11-02__13,36,33"
    # reduced = "perturbation_measurements_2024-11-02__13,19,45"
    # all_terms = "perturbation_measurements_2024-11-02__13,36,33"
    # plot_experiment_comparison(
    #     [diagonal, reduced, all],
    #     ["diagonal", "reduced", "all_terms"],
    # )

    # first second order comparison
    # diagonal = "diagonalization_measurements_2024-11-02__16,32,00"
    # first_order = "perturbation_measurements_2024-11-02__16,39,46"
    # second_order = "perturbation_measurements_2024-11-02__16,32,00"
    # plot_experiment_comparison(
    #     [diagonal, first_order, second_order],
    #     ["diagonal", "pert. first", "pert. second"],
    # )

    # delete summands from second_order array test
    # drop_one = "perturbation_measurements_2024-11-02__17,38,14"
    # drop_two = "perturbation_measurements_2024-11-02__17,32,09"
    # drop_three = "perturbation_measurements_2024-11-02__17,37,34"
    # all_factors = "perturbation_measurements_2024-11-02__17,36,48"
    # plot_experiment_comparison(
    #     [all_factors, drop_one, drop_two, drop_three],
    #     ["all_factors", "drop_one", "drop_two", "drop_three"],
    # )

    # second order optimization end to end
    # diagonal = "diagonalization_measurements_2024-11-03__15,56,10"
    # second_order_optimized = (
    #     "perturbation_measurements_2024-11-03__15,56,10"  # 381.560 seconds
    # )
    # second_order_canonical = (
    #     "perturbation_measurements_2024-11-03__16,05,58"  # 403.665 seconds
    # )
    # second_order_monte_carlo = (
    #     "perturbation_measurements_2024-11-03__16,18,32"  # 3625.690 seconds
    # )
    # plot_experiment_comparison(
    #     [
    #         diagonal,
    #         second_order_optimized,
    #         second_order_canonical,
    #         second_order_monte_carlo,
    #     ],
    #     ["diagonal", "opimized", "canonical", "MC"],
    # )

    # Variational classical networks
    # first_order_perturbation = "2024-11-05__16,13,17-0"
    # second_order_perturbation = "2024-11-07__00,58,28-0"
    # exact_diagonalization = "2024-11-06__14,43,45-0"
    # zeroth_order = "2024-11-07__00,51,22-0"
    # vcn_testem3 = "2024-11-07__01,02,13-0"
    # vcn_testem2 = "2024-11-07__01,19,04-0"
    # latest = get_newest_file_name()
    # plot_experiment_comparison(
    #     [
    #         exact_diagonalization,
    #         zeroth_order,
    #         first_order_perturbation,
    #         second_order_perturbation,
    #         # vcn_testem3,
    #         # vcn_testem2,
    #         # latest,
    #     ],
    #     [
    #         "exact",
    #         "zero_order",
    #         "analytical-o1",
    #         "analytical-o2",
    #         # "vcnem3",
    #         # "vcnem2",
    #         # "latest",
    #     ],
    #     [(0, 1), (0, 2), (0, 3)],
    # )

    # # Concurrence proof of concept verification

    # diagonal = "diagonal"
    # diagonalsampled = "diagonal-sampled"
    # perturbationo0 = "perturbation-o0"
    # perturbationo1 = "perturbation-o1"
    # perturbationo2 = "perturbation-o2"
    # plot_experiment_comparison(
    #     [diagonal, diagonalsampled, perturbationo0, perturbationo1, perturbationo2],
    # )

    # Energy for real
    diagonalization = "diagonalization_measurements_2024-11-26__11,44,27"
    plot_experiment_comparison(
        [diagonalization],
    )
