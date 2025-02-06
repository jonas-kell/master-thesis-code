import os
from typing import Any, List


def main():
    # !! use to configure parameter

    experiment_name_env_var_name = "EXPERIMENT_TASK_NAME"
    experiment_name_env_value = os.getenv(experiment_name_env_var_name)

    if experiment_name_env_value is None:
        print(f"Environment variable {experiment_name_env_var_name} is not set.")
        return

    print(f"Prepare Batch run of experiment: {experiment_name_env_value}")

    if experiment_name_env_value == "concurrence_from_spin":
        # needs: --array=0-0
        experiment = "concurrence_from_spin"
        parameter_array: List[Any] = [0]

    elif experiment_name_env_value == "j_sweep":
        # needs: --array=0-6
        experiment = "j_sweep"
        parameter_array: List[Any] = [
            # param*U here is J
            0.1,
            0.09,
            0.08,
            0.07,
            0.06,
            0.04,
            0.02,
        ]

    elif experiment_name_env_value == "monte_carlo_variance_test":
        # needs: --array=0-4
        experiment = "monte_carlo_variance_test"
        parameter_array: List[Any] = [
            # param is here num mc-samples
            400,
            2000,
            4000,
            20000,
            40000,
        ]

    elif experiment_name_env_value == "variational_classical_networks":
        # needs: --array=0-4
        experiment = "variational_classical_networks"
        parameter_array: List[Any] = [
            # param is here effective time-steps in 1/U
            0,  # this does the exact calculations
            6e-2,
            2e-2,
            6e-3,
            2e-3,
        ]

    elif experiment_name_env_value == "seed_and_init_spread":
        # needs: --array=0-4
        experiment = "seed_and_init_spread"
        parameter_array: List[Any] = [
            # param here controls the seed
            1.1,
            2.2,
            3.3,
            4.4,
            5.5,
        ]

    elif experiment_name_env_value == "energy_behavior":
        # needs: --array=0-3
        experiment = "energy_behavior"
        parameter_array: List[Any] = [
            # param here controls the model to compare
            0,  # o0
            1,  # o1
            2,  # o2
            3,  # exact
        ]

    elif experiment_name_env_value == "square_vcn_comparison":
        # needs: --array=0-4
        experiment = "square_vcn_comparison"
        parameter_array: List[Any] = [
            # param here controls the model to compare
            0,  # vcn - analytical
            1,  # vcn - step mult
            50,  # vcn - step mult
            100,  # vcn - step mult
            200,  # vcn - step mult
        ]

    else:
        print(
            f"Environment variable {experiment_name_env_var_name} has non-allowed value: {experiment_name_env_value}"
        )
        return

    # !! use to configure parameter
    print(
        f"Assignment Script contains the Job Description for {len(parameter_array)} jobs"
    )

    task_id_env_var_name = "SLURM_ARRAY_TASK_ID"
    task_id_env_value = os.getenv(task_id_env_var_name)

    if task_id_env_value is None:
        print(f"Environment variable {task_id_env_var_name} is not set.")
        return

    try:
        # Parse the value to an integer
        task_id_int_value = int(task_id_env_value)
    except ValueError:
        print(f"Cannot convert {task_id_env_value} to an integer.")
        return

    task_count_env_var_name = "SLURM_ARRAY_TASK_COUNT"
    task_count_env_value = os.getenv(task_count_env_var_name)

    if task_count_env_value is None:
        print(f"Environment variable {task_count_env_var_name} is not set.")
        return

    try:
        # Parse the value to an integer
        task_count_int_value = int(task_count_env_value)
    except ValueError:
        print(f"Cannot convert {task_count_env_value} to an integer.")
        return

    num_threads_env_var_name = "SLURM_CPUS_PER_TASK"
    num_threads_env_value = os.getenv(num_threads_env_var_name)

    if num_threads_env_value is None:
        print(f"Environment variable {num_threads_env_var_name} is not set.")
        return

    try:
        # Parse the value to an integer
        num_threads_int_value = int(num_threads_env_value)
    except ValueError:
        print(f"Cannot convert {num_threads_env_value} to an integer.")
        return

    # Running the main script
    print(f"Task id {task_id_int_value} of {task_count_int_value} jobs")

    relevant_parameters = [
        parameter
        for index, parameter in enumerate(parameter_array)
        if index % task_count_int_value == task_id_int_value
    ]

    if len(relevant_parameters) == 0:
        print("Nothing to do for this task runner in this case")
        print(
            "(If you wait long to get the number of tasks on your server, try setting the requested number of jobs more precisely)"
        )
        return

    # shell out calling the main script
    for parameter in relevant_parameters:
        print(f"Shelling out to script run with parameter {parameter}")

        python_executable = "python"
        os.system(
            f"{python_executable} ./../../calculation-helpers/aggregator.py --is_hpc --hpc_task_id {task_id_int_value} --number_workers {num_threads_int_value} --parameter {parameter} --experiment {experiment}"
        )

    if len(relevant_parameters) == 0:
        print(f"no relevant parameters for this runner with id {task_id_int_value}")
        print(parameter_array)


if __name__ == "__main__":
    main()
