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


def run_experiment(data, is_hpc: bool, record_hamiltonian_properties: bool):
    arguments_string = "--do_not_plot do_not_plot"
    for key, val in data.items():
        arguments_string += " --" + key + " " + str(val)

    python_executable = "python"
    os.system(
        f"{python_executable} ./../{'../' if is_hpc else ''}computation-scripts/script.py {arguments_string} {'--record_hamiltonian_properties' if record_hamiltonian_properties else ''}"
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
    ] = cast(
        str, get_argument(args, "experiment", str, "variational_classical_networks")
    )
    plotting = True

    print("Running aggregator script for experiment:", experiment)

    seed_string = "experiment_main_seed"

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
            ("base_energy_only", "o0"),
            ("both_optimizations", "o1"),
            ("both_optimizations_second_order", "o2"),
        ]

        variational_step_fraction_multiplier = 100
        init_sigma = 0.0001
        pseudo_inverse_cutoff = 1e-10  # not switched on

        record_hamiltonian_properties: bool = False
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
            ("base_energy_only", "o0"),
            ("both_optimizations", "o1"),
            ("both_optimizations_second_order", "o2"),
        ]

        variational_step_fraction_multiplier = 100  # not switched on
        init_sigma = 0.0001  # not switched on

        record_hamiltonian_properties: bool = False
        observable_set = "concurrence_and_pauli"
        pseudo_inverse_cutoff = 1e-10  # not switched on

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
            ("base_energy_only", "o0"),
            ("both_optimizations", "o1"),
            ("both_optimizations_second_order", "o2"),
        ]

        variational_step_fraction_multiplier = 100  # not switched on
        init_sigma = 0.0001  # not switched on
        pseudo_inverse_cutoff = 1e-10  # not switched on

        record_hamiltonian_properties: bool = False
        observable_set = "current_and_occupation"

        scaler = 50
        num_samples_over_timespan = 16
        target_time_in_1_over_u = scaler * num_samples_over_timespan

        zip_filename_base = str(num_monte_carlo_samples) + "samples"

    elif experiment == "energy_behavior":
        # ! energy-behavior
        U = 1.0
        E = 2.5
        J = 0.1
        n = 4
        phi = 0.1
        system_geometry_type = "chain"

        num_monte_carlo_samples = 1  # not switched on
        num_samples_per_chain = num_monte_carlo_samples // 10

        do_exact_diagonalization = True
        do_exact_comparison = True
        different_monte_carlo_tests = 0  # not switched on

        compare_type_hamiltonians = [
            ("base_energy_only", "o0"),
            ("both_optimizations", "o1"),
            ("both_optimizations_second_order", "o2"),
        ]

        variational_step_fraction_multiplier = 100  # not switched on
        init_sigma = 0.0001  # not switched on
        pseudo_inverse_cutoff = 1e-10  # not switched on

        record_hamiltonian_properties: bool = False
        observable_set = "energy_and_variance"

        scaler = 1 / J
        num_samples_over_timespan = 150
        target_time_in_1_over_u = scaler * 1.25

        zip_filename_base = "energy-variance"

    elif experiment == "variational_classical_networks":
        # ! energy-behavior
        U = 1.0
        E = 2.5
        J = 0.05
        n = 3
        phi = 0.1
        system_geometry_type = "chain"

        num_monte_carlo_samples = 1  # not switched on
        num_samples_per_chain = num_monte_carlo_samples // 10

        do_exact_diagonalization = False
        do_exact_comparison = True
        different_monte_carlo_tests = 0  # not switched on

        compare_type_hamiltonians = [
            ("variational_classical_networks", "vcn"),
            ("variational_classical_networks_analytical_factors", "vcn_analytical"),
        ]

        effective_timestep_step_in_1_over_u = 2 * 1e-4
        init_sigma = 0.0001
        pseudo_inverse_cutoff = 1e-10

        record_hamiltonian_properties: bool = True
        observable_set = "energy_and_variance"

        scaler = 1 / U
        num_samples_over_timespan = 50
        target_time_in_1_over_u = scaler * 3

        zip_filename_base = "vcn-param-tests"

        # the step fraction multiplier is calculated in this experiment
        variational_step_fraction_multiplier = int(
            np.ceil(
                (target_time_in_1_over_u / (num_samples_over_timespan))
                / effective_timestep_step_in_1_over_u
            )
        )
        print(
            "Calculated variational_step_fraction_multiplier of:",
            variational_step_fraction_multiplier,
        )
    else:
        raise Exception("Unknown Experiment Specification")

    # !! compute experiment settings below this
    num_multithread_workers = cast(int, get_argument(args, "number_workers", int, 6))

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
    }

    if do_exact_diagonalization:
        exact_name = run_file_name_base + "-exact-exact"
        file_names_list.append(exact_name)
        experiment_data = {
            "number_workers": 1,  # this is faster without multithreading
            "hamiltonian_type": "exact",
            "file_name_overwrite": exact_name,
            "sampling_strategy": "exact",
            **experiment_base_data,
        }
        run_experiment(experiment_data, is_hpc, record_hamiltonian_properties)

    if do_exact_comparison:
        for compare_type_hamiltonian, order_slug in compare_type_hamiltonians:
            compare_exact_name = run_file_name_base + f"-compare{order_slug}-exact"
            file_names_list.append(compare_exact_name)
            experiment_data = {
                "hamiltonian_type": compare_type_hamiltonian,
                "file_name_overwrite": compare_exact_name,
                "sampling_strategy": "exact",
                "number_workers": num_multithread_workers,
                **experiment_base_data,
            }
            run_experiment(experiment_data, is_hpc, record_hamiltonian_properties)

    for i in range(different_monte_carlo_tests):
        for compare_type_hamiltonian, order_slug in compare_type_hamiltonians:
            mc_name = run_file_name_base + f"-compare{order_slug}-mc{i}"
            file_names_list.append(mc_name)
            experiment_data = {
                "hamiltonian_type": compare_type_hamiltonian,
                "file_name_overwrite": mc_name,
                "randomness_seed": generate_random_string(),
                "num_samples_per_chain": num_samples_per_chain,
                "num_monte_carlo_samples": num_monte_carlo_samples,
                "sampling_strategy": "monte_carlo",
                "number_workers": num_multithread_workers,
                **experiment_base_data,
            }
            run_experiment(experiment_data, is_hpc, record_hamiltonian_properties)

    zip_filename = f"{run_file_name_base}.zip"
    print(zip_filename)
    zip_files(file_names_list, zip_filename)

    if plotting and not is_hpc:
        plot_experiment_comparison(file_names_list)


if __name__ == "__main__":
    main()
