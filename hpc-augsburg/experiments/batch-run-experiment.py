import os
from typing import Dict, Any, List


def main():

    # !! use to configure spread parameters
    experiments_array: List[Dict[Any, Any]] = [
        {"U": 0.1, "E": -0.1},
        {"U": 0.2, "E": -0.2},
        {"U": 0.3, "E": -0.3},
        {"U": 0.4, "E": -0.4},
    ]
    print(
        f"Assignment Script contains the Job Description for {len(experiments_array)} jobs"
    )
    # !! use to configure spread parameters

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

    relevant_experiments = [
        experiment
        for index, experiment in enumerate(experiments_array)
        if index % task_count_int_value == task_id_int_value
    ]

    # shell out calling the main script
    for experiment in relevant_experiments:
        print(f"Shelling out to script run with parameters {experiment}")

        arguments_string = f"--job_array_index {task_id_int_value} --number_workers {num_threads_int_value}"
        for key, val in experiment.items():
            arguments_string += " --" + key + " " + str(val)

        python_executable = "python"
        os.system(
            f"{python_executable} ./../../computation-scripts/script.py {arguments_string}"
        )


if __name__ == "__main__":
    main()
