import os
from typing import Any, List


def main():
    # !! use to configure parameter
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
    print(
        f"Assignment Script contains the Job Description for {len(parameter_array)} jobs"
    )
    # !! use to configure parameter

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

    # shell out calling the main script
    for parameter in relevant_parameters:
        print(f"Shelling out to script run with parameter {parameter}")

        python_executable = "python"
        os.system(
            f"{python_executable} ./../../calculation-helpers/aggregator.py --is_hpc --hpc_task_id {task_id_int_value} --number_workers {num_threads_int_value} --parameter {parameter}"
        )


if __name__ == "__main__":
    main()
