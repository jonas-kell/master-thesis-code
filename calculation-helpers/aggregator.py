from typing import cast, Dict, Union, Type
import os
import random
import hashlib
import string
from datetime import datetime
from plotcompare import plot_experiment_comparison
import zipfile
from commonsettings import get_full_file_path
import argparse

AcceptableTypes = Union[Type[int], Type[float], Type[str]]


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


def run_experiment(data, is_hpc: bool):
    arguments_string = "--do_not_plot do_not_plot"
    for key, val in data.items():
        arguments_string += " --" + key + " " + str(val)

    python_executable = "python"
    os.system(
        f"{python_executable} ./../{'../' if is_hpc else ''}computation-scripts/script.py {arguments_string}"
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
    parser.add_argument("--parameter", required=False)
    parser.add_argument("--is_hpc", action="store_true", required=False)
    parser.add_argument("--hpc_task_id", required=False)
    args = vars(parser.parse_args())
    is_hpc = args["is_hpc"]
    hpc_task_id = cast(int, get_argument(args, "hpc_task_id", int, 0))
    # ! arg parse section end

    print("Running aggregator script")

    seed_string = "experiment_main_seed"

    parameter = cast(int, get_argument(args, "parameter", int, 0))

    U = 1.0
    E = 2.5
    J = 0.1
    n = 4
    phi = 0.1

    num_monte_carlo_samples = parameter
    num_samples_per_chain = num_monte_carlo_samples // 10

    different_monte_carlo_tests = 10

    scaler = 8
    # goal: for one of the smaller J=0.01U this is t=scaler*J, but we calc in U, because that is constant when we do runs in J
    target_time_in_1_over_u = scaler * 100
    num_samples_over_timespan = 2 * scaler

    # compare_type_hamiltonian = "base_energy_only"
    # compare_type_hamiltonian = "both_optimizations"
    compare_type_hamiltonian = "both_optimizations_second_order"

    plotting = True

    # !! compute experiment settings below this
    num_multithread_workers = cast(int, get_argument(args, "number_workers", int, 6))

    seed_random_generator(seed_string)
    run_file_name_base = (
        "aggregator-"
        + ((hpc_task_id + "-") if is_hpc else "")
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
    }

    exact_name = run_file_name_base + "-exact-exact"
    file_names_list.append(exact_name)
    experiment_data = {
        "number_workers": 1,  # this is faster without multithreading
        "hamiltonian_type": "exact",
        "file_name_overwrite": exact_name,
        "sampling_strategy": "exact",
        **experiment_base_data,
    }
    run_experiment(experiment_data, is_hpc)

    compare_exact_name = run_file_name_base + "-compare-exact"
    file_names_list.append(compare_exact_name)
    experiment_data = {
        "hamiltonian_type": compare_type_hamiltonian,
        "file_name_overwrite": compare_exact_name,
        "sampling_strategy": "exact",
        "number_workers": num_multithread_workers,
        **experiment_base_data,
    }
    run_experiment(experiment_data, is_hpc)

    for i in range(different_monte_carlo_tests):
        mc_name = run_file_name_base + f"-compare-mc{i}"
        file_names_list.append(mc_name)
        experiment_data = {
            "hamiltonian_type": compare_type_hamiltonian,
            "file_name_overwrite": mc_name,
            "randomnes_seed": generate_random_string(),
            "num_samples_per_chain": num_samples_per_chain,
            "num_monte_carlo_samples": num_monte_carlo_samples,
            "sampling_strategy": "monte_carlo",
            "number_workers": num_multithread_workers,
            **experiment_base_data,
        }
        run_experiment(experiment_data, is_hpc)

    zip_filename = f"{run_file_name_base}.zip"
    print(zip_filename)
    zip_files(file_names_list, zip_filename)

    if plotting:
        plot_experiment_comparison(file_names_list)


if __name__ == "__main__":
    main()
