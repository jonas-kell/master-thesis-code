import os
import random
import hashlib
import string
from datetime import datetime
from plotcompare import plot_experiment_comparison
import zipfile
from commonsettings import get_full_file_path


def seed_random_generator(global_seed):
    seed = int(hashlib.sha256(global_seed.encode("utf-8")).hexdigest(), 16)
    random.seed(seed)


def generate_random_string(length=8):
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for _ in range(length))


def run_experiment(data):
    arguments_string = "--do_not_plot do_not_plot"
    for key, val in data.items():
        arguments_string += " --" + key + " " + str(val)

    python_executable = "python"
    os.system(
        f"{python_executable} ./../computation-scripts/script.py {arguments_string}"
    )


def zip_files(file_names, zip_name):
    path_with_json = get_full_file_path(zip_name)
    with zipfile.ZipFile(path_with_json.replace(".json", ""), "w") as zipf:
        for file_name in file_names:
            file_full_path = get_full_file_path(file_name)
            zipf.write(file_full_path, arcname=file_full_path.split("/")[-1])


def main():
    seed_string = "experiment_main_seed"

    U = 1.0
    E = 2.5
    J = 0.1
    n = 4
    phi = 0.1

    num_monte_carlo_samples = 400
    num_samples_per_chain = num_monte_carlo_samples // 10

    different_monte_carlo_tests = 5

    scaler = 8
    # goal: for one of the smaller J=0.01U this is t=scaler*J, but we calc in U, because that is constant when we do runs in J
    target_time_in_1_over_u = scaler * 100
    num_samples_over_timespan = 2 * scaler

    # compare_type_hamiltonian = "base_energy_only"
    # compare_type_hamiltonian = "both_optimizations"
    compare_type_hamiltonian = "both_optimizations_second_order"

    multithread_num_workers = 6

    plotting = True

    # !! compute experiment settings below this

    seed_random_generator(seed_string)
    run_file_name_base = "aggregator-" + datetime.now().strftime("%Y-%m-%d__%H,%M,%S")

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
    run_experiment(experiment_data)

    compare_exact_name = run_file_name_base + "-compare-exact"
    file_names_list.append(compare_exact_name)
    experiment_data = {
        "number_workers": multithread_num_workers,
        "hamiltonian_type": compare_type_hamiltonian,
        "file_name_overwrite": compare_exact_name,
        "sampling_strategy": "exact",
        **experiment_base_data,
    }
    run_experiment(experiment_data)

    for i in range(different_monte_carlo_tests):
        mc_name = run_file_name_base + f"-compare-mc{i}"
        file_names_list.append(mc_name)
        experiment_data = {
            "number_workers": multithread_num_workers,
            "hamiltonian_type": compare_type_hamiltonian,
            "file_name_overwrite": mc_name,
            "randomnes_seed": generate_random_string(),
            "num_samples_per_chain": num_samples_per_chain,
            "num_monte_carlo_samples": num_monte_carlo_samples,
            "sampling_strategy": "monte_carlo",
            **experiment_base_data,
        }
        run_experiment(experiment_data)

    zip_filename = f"{run_file_name_base}.zip"
    print(zip_filename)
    zip_files(file_names_list, zip_filename)

    if plotting:
        plot_experiment_comparison(file_names_list)


if __name__ == "__main__":
    main()
