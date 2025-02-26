from typing import cast, Dict, Union, Type, Literal
import os
import random
import hashlib
import string
from datetime import datetime
from plotcompare import plot_experiment_comparison
import zipfile
from commonsettings import get_full_file_path
import argparse
import decimal
import numpy as np

AcceptableTypes = Union[Type[int], Type[float], Type[str]]


def float_to_str(f):
    ctx = decimal.Context()
    ctx.prec = 20
    d1 = ctx.create_decimal(repr(f))
    return format(d1, "f")


def get_argument(
    arguments: Dict[str, Union[float, int, str]],
    name: str,
    acc_type: AcceptableTypes,
    default: Union[float, int, str],
):
    extracted_value = default
    try:
        if arguments[name] is None:
            raise ValueError("None set Value")
        extracted_value = acc_type(arguments[name])
        print(f"Using non-default Argument for param {name}: {extracted_value}")
    except ValueError:
        pass
    except TypeError:
        pass

    return extracted_value


def seed_random_generator(global_seed):
    seed = int(hashlib.sha256(global_seed.encode("utf-8")).hexdigest(), 16)
    random.seed(seed)


def generate_random_string(length=8):
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for _ in range(length))


def run_experiment(
    data,
    is_hpc: bool,
    record_hamiltonian_properties: bool,
    record_imag_part: bool,
    ue_might_change: bool,
    ue_variational: bool,
):
    arguments_string = "--do_not_plot do_not_plot"
    for key, val in data.items():
        arguments_string += " --" + key + " " + str(val)

    python_executable = "python"
    os.system(
        f"{python_executable} ./../{'../' if is_hpc else ''}computation-scripts/script.py {arguments_string} {'--record_hamiltonian_properties' if record_hamiltonian_properties else ''} {'--record_imag_part' if record_imag_part else ''} {'--ue_might_change' if ue_might_change else ''}  {'--ue_variational' if ue_variational else ''}"
    )


def zip_files(file_names, zip_name):
    path_with_json = get_full_file_path(zip_name)
    with zipfile.ZipFile(path_with_json.replace(".json", ""), "w") as zipf:
        for file_name in file_names:
            file_full_path = get_full_file_path(file_name)
            zipf.write(file_full_path, arcname=file_full_path.split("/")[-1])


def main():
    # ! arg parse section
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--number_workers", required=False)
    parser.add_argument("--experiment", required=False)
    parser.add_argument("--parameter", required=False)
    parser.add_argument("--is_hpc", action="store_true", required=False)
    parser.add_argument("--hpc_task_id", required=False)
    args = vars(parser.parse_args())
    is_hpc = args["is_hpc"]
    hpc_task_id = cast(int, get_argument(args, "hpc_task_id", int, 0))
    # ! arg parse section end

    experiment: Literal[
        "j_sweep",
        "concurrence_from_spin",
        "monte_carlo_variance_test",
        "energy_behavior",
        "variational_classical_networks",
        "seed_and_init_spread",
        "square_vcn_small",
        "square_vcn_comparison",
        "energy_conservation",
        "system_size_dependency",
    ] = cast(str, get_argument(args, "experiment", str, "system_size_dependency"))
    plotting = True

    print("Running aggregator script for experiment:", experiment)

    seed_string = "experiment_main_seed"
    randomness_seed = "very_nice_seed"

    # psi selection type might get overwritten by the experiment
    psi_selection_type: Literal["chain_canonical", "chain_combined"] = "chain_canonical"

    # worker number might get overwritten by the experiment
    num_multithread_workers = cast(int, get_argument(args, "number_workers", int, 6))
    # Only necessary for VCN experimnts
    ue_might_change = False
    ue_variational = False

    # !! configure experiment settings below this
    if experiment == "j_sweep":
        parameter = cast(
            float, get_argument(args, "parameter", float, 0.1)
        )  # parameter is for giving j scaler
        # ! j-sweep
        U = 1.0
        E = 2.5
        J = parameter * U
        n = 6
        phi = 0.1
        system_geometry_type = "chain"

        num_monte_carlo_samples = 10000
        num_samples_per_chain = num_monte_carlo_samples // 10

        do_exact_diagonalization = True
        do_exact_comparison = True
        different_monte_carlo_tests = 1

        compare_type_hamiltonians = [
            ("zeroth_order_optimized", "o0"),
            ("first_order_optimized", "o1"),
            ("second_order_optimized", "o2"),
        ]

        variational_step_fraction_multiplier = 100
        init_sigma = 0.0001  # not switched on
        vcn_parameter_init_distribution = "normal"  # not switched on
        pseudo_inverse_cutoff = 1e-10  # not switched on

        record_hamiltonian_properties: bool = False
        record_imag_part: bool = False
        observable_set = "current_and_occupation"

        scaler = 1
        target_time_in_1_over_u = scaler * 200
        num_samples_over_timespan = 30

        zip_filename_base = "j" + float_to_str(J).replace(".", "")

    elif experiment == "concurrence_from_spin":
        # ! concurrence-from-spin
        U = 1.0
        E = 0.5
        J = 0.1
        n = 4
        phi = 0.1 * np.pi
        system_geometry_type = "chain"

        num_monte_carlo_samples = 10  # not switched on
        num_samples_per_chain = num_monte_carlo_samples // 10

        do_exact_diagonalization = True
        do_exact_comparison = True
        different_monte_carlo_tests = 0  # not switched on

        compare_type_hamiltonians = [
            ("zeroth_order_optimized", "o0"),
            ("first_order_optimized", "o1"),
            ("second_order_optimized", "o2"),
        ]

        variational_step_fraction_multiplier = 100  # not switched on
        init_sigma = 0.0001  # not switched on
        vcn_parameter_init_distribution = "normal"  # not switched on
        pseudo_inverse_cutoff = 1e-10  # not switched on

        record_hamiltonian_properties: bool = False
        record_imag_part: bool = False
        observable_set = "concurrence_and_pauli"

        scaler = 1 / J
        target_time_in_1_over_u = scaler * 1
        num_samples_over_timespan = 200

        zip_filename_base = "concurrence-comparison"

    elif experiment == "monte_carlo_variance_test":
        parameter = cast(
            int, get_argument(args, "parameter", int, 400)
        )  # parameter is for giving num mc-samples
        # ! monte-carlo-variance-test
        U = 1.0
        E = 2.5
        J = 0.1
        n = 4
        phi = 0.1
        system_geometry_type = "chain"

        num_monte_carlo_samples = parameter
        num_samples_per_chain = num_monte_carlo_samples // 10

        do_exact_diagonalization = False
        do_exact_comparison = True
        different_monte_carlo_tests = 10  # switched on in variance mode

        compare_type_hamiltonians = [
            ("zeroth_order_optimized", "o0"),
            ("first_order_optimized", "o1"),
            ("second_order_optimized", "o2"),
        ]

        variational_step_fraction_multiplier = 100  # not switched on
        init_sigma = 0.0001  # not switched on
        vcn_parameter_init_distribution = "normal"  # not switched on
        pseudo_inverse_cutoff = 1e-10  # not switched on

        record_hamiltonian_properties: bool = False
        record_imag_part: bool = False
        observable_set = "current_and_occupation"

        scaler = 50
        num_samples_over_timespan = 16
        target_time_in_1_over_u = scaler * num_samples_over_timespan

        zip_filename_base = str(num_monte_carlo_samples) + "samples"

    elif experiment == "energy_behavior":
        parameter = cast(
            int, get_argument(args, "parameter", int, 0)
        )  # parameter is for selecting one of the compare hamiltonians (faster calculation)
        # ! energy-behavior

        U = 1.0
        E = 2.5
        J = 0.1
        n = 4  # for n=5 this is basically no longer reasonably computable  100% prognosis: 51628.4s
        phi = 0.1
        system_geometry_type = "chain"

        num_monte_carlo_samples = 1  # not switched on
        num_samples_per_chain = num_monte_carlo_samples // 10

        do_exact_diagonalization = parameter == 3
        do_exact_comparison = parameter != 3  # comparison only on parameter == 3
        different_monte_carlo_tests = 0  # not switched on

        compare_type_hamiltonians = [
            comp
            for index, comp in enumerate(
                [
                    ("zeroth_order_optimized", "o0"),
                    ("first_order_optimized", "o1"),
                    ("second_order_optimized", "o2"),
                ]
            )
            if index
            == parameter  # filter the respective comparison model, if parameter is 0,1,2
        ]

        variational_step_fraction_multiplier = 100  # not switched on
        init_sigma = 0.0001  # not switched on
        vcn_parameter_init_distribution = "normal"  # not switched on
        pseudo_inverse_cutoff = 1e-10  # not switched on

        record_hamiltonian_properties: bool = False
        record_imag_part: bool = False
        observable_set = "energy_and_variance"

        scaler = 1 / J
        num_samples_over_timespan = 150
        target_time_in_1_over_u = scaler * 1.25

        zip_filename_base = f"energy-variance-run{parameter}"

    elif experiment == "seed_and_init_spread":
        parameter = cast(
            float, get_argument(args, "parameter", float, 0.1)
        )  # parameter is for deciding seed

        # overwrites the seed
        randomness_seed = f"{float_to_str(parameter)}{float_to_str(parameter)}{float_to_str(parameter)}"

        # ! test the effective step-size of the variational-classical network
        U = 1.0
        E = 2.5
        J = 0.1
        n = 8
        phi = 0.1
        system_geometry_type = "chain"

        num_monte_carlo_samples = 1  # not switched on
        num_samples_per_chain = num_monte_carlo_samples // 10

        ue_might_change = False
        ue_variational = False

        do_exact_comparison = True
        different_monte_carlo_tests = 0  # not switched on

        do_exact_diagonalization = False
        compare_type_hamiltonians = [
            (
                "variational_classical_networks",
                f"vcn-{float_to_str(parameter).replace('.', '')}",
            ),
        ]

        init_sigma = 0.3
        vcn_parameter_init_distribution = (
            "uniform"  # we want to show different starting values here
        )
        pseudo_inverse_cutoff = 1e-10

        record_hamiltonian_properties: bool = True
        record_imag_part: bool = True
        observable_set = "energy_and_variance"

        scaler = 1 / U
        num_samples_over_timespan = 20
        target_time_in_1_over_u = scaler * 1.5

        variational_step_fraction_multiplier = 3
        zip_filename_base = (
            f"vcn-init-tests-{float_to_str(init_sigma).replace('.', '')}"
        )

    elif experiment == "variational_classical_networks":
        parameter = cast(
            float, get_argument(args, "parameter", float, 0)
        )  # parameter is for deciding effective_timestep_step_in_1_over_u
        effective_timestep_step_in_1_over_u = parameter

        # ! test the effective step-size of the variational-classical network
        U = 1.0
        E = 2.5
        J = 0.1
        n = 6
        phi = 0.1
        system_geometry_type = "chain"

        num_monte_carlo_samples = 1  # not switched on
        num_samples_per_chain = num_monte_carlo_samples // 10

        ue_might_change = False
        ue_variational = False

        do_exact_comparison = True
        different_monte_carlo_tests = 0  # not switched on

        do_exact_diagonalization = False  # for energy and variance we know the t=0 values are correct, therefore useless to compute exact diagonalization measurements
        if parameter == 0:
            # do the "exact" comparisons
            compare_type_hamiltonians = [
                ("zeroth_order_optimized", "o0"),
                ("first_order_optimized", "o1"),
                ("second_order_optimized", "o2"),
                ("variational_classical_networks_analytical_factors", "vcnanalytical"),
            ]
        else:
            # do the VCN EXPERIMENT
            compare_type_hamiltonians = [
                (
                    "variational_classical_networks",
                    f"vcn{float_to_str(effective_timestep_step_in_1_over_u).replace('.', '')}",
                ),
            ]

        init_sigma = 0.0001
        pseudo_inverse_cutoff = 1e-10
        vcn_parameter_init_distribution = "normal"  # not switched on

        record_hamiltonian_properties: bool = True
        record_imag_part: bool = True
        observable_set = "energy_and_variance"

        scaler = 1 / U
        num_samples_over_timespan = 50
        target_time_in_1_over_u = scaler * 3

        # the step fraction multiplier is calculated in this experiment
        if parameter != 0:
            variational_step_fraction_multiplier = int(
                np.ceil(
                    (target_time_in_1_over_u / (num_samples_over_timespan))
                    / effective_timestep_step_in_1_over_u
                )
            )
            zip_filename_base = f"vcn-param-tests-ets{float_to_str(effective_timestep_step_in_1_over_u).replace('.', '')}"
        else:
            variational_step_fraction_multiplier = 1  # is deactiavted
            zip_filename_base = "vcn-param-tests-exact"

        print(
            "Calculated variational_step_fraction_multiplier of:",
            variational_step_fraction_multiplier,
        )

    elif experiment == "square_vcn_small":
        parameter = cast(
            int, get_argument(args, "parameter", int, 0)
        )  # parameter is for setting number of intermediate steps

        # ! test a complete square vcn comparison
        U = 1.0
        E = 0.8
        J = 0.1
        n = 2
        phi = np.pi * 0.6
        system_geometry_type = "square"

        num_monte_carlo_samples = 1  # not switched on
        num_samples_per_chain = num_monte_carlo_samples // 10

        ue_might_change = True
        ue_variational = True

        do_exact_comparison = True
        different_monte_carlo_tests = 0  # not switched on

        do_exact_diagonalization = False  # for energy and variance we know the t=0 values are correct, therefore useless to compute exact diagonalization measurements
        if parameter == 0:
            compare_type_hamiltonians = [
                ("variational_classical_networks_analytical_factors", "vcnanalytical"),
                (
                    "first_order_optimized",
                    "o1",
                ),  # compare energy for programming correctness
                (
                    "second_order_optimized",
                    "o2",
                ),  # show how much better vcn is (hopefully)
            ]
        else:
            compare_type_hamiltonians = [
                ("variational_classical_networks", f"vcn{parameter}"),
            ]

        init_sigma = 0.00001
        vcn_parameter_init_distribution = "normal"
        pseudo_inverse_cutoff = 1e-10

        record_hamiltonian_properties: bool = True
        record_imag_part: bool = True
        observable_set = "energy_and_variance"

        steps = 50
        num_samples_over_timespan = steps + 1
        target_time_in_1_over_u = 1 / U * 25

        # computed, works with "chain..." and "square..."
        psi_selection_type = system_geometry_type + "_canonical"

        variational_step_fraction_multiplier = parameter  # this now does things!!
        print(
            "Uses variational_step_fraction_multiplier of:",
            variational_step_fraction_multiplier,
        )

        zip_filename_base = "square-vcn-" + (
            "comparisons"
            if parameter == 0
            else f"variational-{variational_step_fraction_multiplier}"
        )

    elif experiment == "square_vcn_comparison":
        parameter = cast(
            int, get_argument(args, "parameter", int, 0)
        )  # parameter is for setting number of intermediate steps

        # ! test a complete square vcn comparison
        U = 1.0
        E = 0.8
        J = 0.1
        n = 3
        phi = np.pi * 0.6
        system_geometry_type = "square"

        num_monte_carlo_samples = 1  # not switched on
        num_samples_per_chain = num_monte_carlo_samples // 10

        ue_might_change = True
        ue_variational = True

        do_exact_comparison = True
        different_monte_carlo_tests = 0  # not switched on

        do_exact_diagonalization = False  # for energy and variance we know the t=0 values are correct, therefore useless to compute exact diagonalization measurements
        if parameter == 0:
            compare_type_hamiltonians = [
                ("variational_classical_networks_analytical_factors", "vcnanalytical"),
                (
                    "first_order_optimized",
                    "o1",
                ),  # compare energy for programming correctness
                (
                    "second_order_optimized",
                    "o2",
                ),  # show how much better vcn is (hopefully)
            ]
        else:
            compare_type_hamiltonians = [
                ("variational_classical_networks", f"vcn{parameter}"),
            ]

        init_sigma = 0.00001
        vcn_parameter_init_distribution = "normal"
        pseudo_inverse_cutoff = 1e-10

        record_hamiltonian_properties: bool = True
        record_imag_part: bool = True
        observable_set = "energy_and_variance"

        steps = 40

        scaler = 1 / U
        num_samples_over_timespan = steps + 1
        target_time_in_1_over_u = scaler * 0.35 * (steps + 1)

        # computed, works with "chain..." and "square..."
        psi_selection_type = system_geometry_type + "_canonical"

        variational_step_fraction_multiplier = parameter  # this now does things!!
        print(
            "Uses variational_step_fraction_multiplier of:",
            variational_step_fraction_multiplier,
        )

        zip_filename_base = "square-vcn-" + (
            "comparisons"
            if parameter == 0
            else f"variational-{variational_step_fraction_multiplier}"
        )

    elif experiment == "energy_conservation":
        parameter = cast(
            int, get_argument(args, "parameter", int, 0)
        )  # parameter is for setting number of intermediate steps
        second_parameter = parameter % 10000  # the one parameter encodes two params

        # ! test the energy is conserved in the beginning
        # (and that it is conserved better, the smaller the effective step-size is)
        U = 1.0
        E = 0.8
        J = 0.1
        phi = np.pi * 0.8

        # chain: n=4
        # square: n=2
        n = parameter // 10000  # the one parameter encodes two params
        system_geometry_type = (
            "chain" if n > 3 else "square"
        )  # small sizes may be square, larger only chain

        num_monte_carlo_samples = 1  # not switched on
        num_samples_per_chain = num_monte_carlo_samples // 10

        ue_might_change = True
        ue_variational = True

        do_exact_comparison = True
        different_monte_carlo_tests = 0  # not switched on

        do_exact_diagonalization = False  # for energy and variance we know the t=0 values are correct, therefore useless to compute exact diagonalization measurements
        if second_parameter == 0:
            compare_type_hamiltonians = [
                (
                    "variational_classical_networks_analytical_factors",
                    f"vcnanalytical_{system_geometry_type}",
                ),
                ("first_order_optimized", f"o1_{system_geometry_type}"),
                ("second_order_optimized", f"o2_{system_geometry_type}"),
            ]
        else:
            compare_type_hamiltonians = [
                (
                    "variational_classical_networks",
                    f"vcn_{system_geometry_type}_{second_parameter}",
                ),
            ]

        init_sigma = 0.00001
        vcn_parameter_init_distribution = "normal"
        pseudo_inverse_cutoff = 1e-10

        record_hamiltonian_properties: bool = True
        record_imag_part: bool = True
        observable_set = "energy_and_variance"

        # chain: 30
        # square: 30
        steps = 30

        num_samples_over_timespan = steps + 1
        # square 3 -> breaks at 1.5 U -> but is generally fine, so same range overall, second step is there -> 10
        # chain 10 -> breaks at ?? U
        target_time_in_1_over_u = 10 / U

        # computed, works with "chain..." and "square..."
        psi_selection_type = system_geometry_type + "_canonical"

        variational_step_fraction_multiplier = second_parameter  # generally grows eponentially to make a differene to the time-step in later sizes

        zip_filename_base = f"energy-conservation-{variational_step_fraction_multiplier}-{system_geometry_type}"

    elif experiment == "system_size_dependency":

        parameter = cast(
            int, get_argument(args, "parameter", int, 2)
        )  # parameter is for setting system size

        # ! test the energy is conserved in the beginning
        # (and that it is conserved better, the smaller the effective step-size is)
        U = 1.0
        E = 0.8
        J = 0.1
        phi = np.pi * 0.8

        n = parameter
        system_geometry_type = "square"

        num_monte_carlo_samples = 30000
        num_samples_per_chain = 1000

        ue_might_change = True
        ue_variational = True

        do_exact_comparison = False
        different_monte_carlo_tests = 1

        do_exact_diagonalization = False  # for energy and variance we know the t=0 values are correct, therefore useless to compute exact diagonalization measurements
        compare_type_hamiltonians = [
            (
                "variational_classical_networks",
                f"vcn_size{n}",
            ),
        ]

        init_sigma = 0.00001
        vcn_parameter_init_distribution = "normal"
        pseudo_inverse_cutoff = 1e-10

        record_hamiltonian_properties: bool = False
        record_imag_part: bool = True
        observable_set = "energy_and_variance_and_entanglement_test"

        steps = 100

        num_samples_over_timespan = steps + 1
        target_time_in_1_over_u = 25 / U

        # computed, works with "chain..." and "square..."
        psi_selection_type = system_geometry_type + "_canonical"

        variational_step_fraction_multiplier = 20

        zip_filename_base = f"system-size-dependency-size{n}"

    else:
        raise Exception("Unknown Experiment Specification")

    # !! compute experiment settings below this
    seed_random_generator(seed_string)
    run_file_name_base = (
        zip_filename_base
        + "-"
        + ((str(hpc_task_id) + "-") if is_hpc else "")
        + datetime.now().strftime("%Y-%m-%d__%H,%M,%S")
    )

    file_names_list = []

    experiment_base_data = {
        "U": U,
        "E": E,
        "J": J,
        "n": n,
        "phi": phi,
        "start_time": 0,
        "number_of_time_steps": num_samples_over_timespan,
        "target_time_in_one_over_u": target_time_in_1_over_u,
        "variational_step_fraction_multiplier": variational_step_fraction_multiplier,
        "init_sigma": init_sigma,
        "system_geometry_type": system_geometry_type,
        "observable_set": observable_set,
        "pseudo_inverse_cutoff": pseudo_inverse_cutoff,
        "randomness_seed": randomness_seed,
        "psi_selection_type": psi_selection_type,
        "vcn_parameter_init_distribution": vcn_parameter_init_distribution,
    }

    if do_exact_diagonalization:
        exact_name = run_file_name_base + "-exact-exact"
        file_names_list.append(exact_name)
        experiment_data = {
            **experiment_base_data,
            "number_workers": 1,  # this is faster without multithreading
            "hamiltonian_type": "exact",
            "file_name_overwrite": exact_name,
            "sampling_strategy": "exact",
        }
        run_experiment(
            experiment_data,
            is_hpc,
            record_hamiltonian_properties,
            record_imag_part,
            ue_might_change,
            ue_variational,
        )

    if do_exact_comparison:
        for compare_type_hamiltonian, order_slug in compare_type_hamiltonians:
            compare_exact_name = run_file_name_base + f"-compare{order_slug}-exact"
            file_names_list.append(compare_exact_name)
            experiment_data = {
                **experiment_base_data,
                "hamiltonian_type": compare_type_hamiltonian,
                "file_name_overwrite": compare_exact_name,
                "sampling_strategy": "exact",
                "number_workers": num_multithread_workers,
            }
            run_experiment(
                experiment_data,
                is_hpc,
                record_hamiltonian_properties,
                record_imag_part,
                ue_might_change,
                ue_variational,
            )

    for i in range(different_monte_carlo_tests):
        for compare_type_hamiltonian, order_slug in compare_type_hamiltonians:
            mc_name = run_file_name_base + f"-compare{order_slug}-mc{i}"
            file_names_list.append(mc_name)
            experiment_data = {
                **experiment_base_data,
                "hamiltonian_type": compare_type_hamiltonian,
                "file_name_overwrite": mc_name,
                "randomness_seed": generate_random_string(),
                "num_samples_per_chain": num_samples_per_chain,
                "num_monte_carlo_samples": num_monte_carlo_samples,
                "sampling_strategy": "monte_carlo",
                "number_workers": num_multithread_workers,
            }
            run_experiment(
                experiment_data,
                is_hpc,
                record_hamiltonian_properties,
                record_imag_part,
                ue_might_change,
                ue_variational,
            )

    zip_filename = f"{run_file_name_base}.zip"
    print(zip_filename)
    zip_files(file_names_list, zip_filename)

    if plotting and not is_hpc:
        plot_experiment_comparison(file_names_list)


if __name__ == "__main__":
    main()
