import os


def main():
    env_var_name = "SLURM_ARRAY_TASK_ID"
    env_value = os.getenv(env_var_name)

    if env_value is None:
        print(f"Environment variable {env_var_name} is not set.")
        return

    try:
        # Parse the value to an integer
        int_value_task_id = int(env_value)
    except ValueError:
        print(f"Cannot convert {env_value} to an integer.")
        return

    # Running the main script
    print(int_value_task_id)  # TODO use to spread parameters

    # shell out calling the main script
    python_executable = "python"
    os.system(f"{python_executable} ./../../computation-scripts/script.py")


if __name__ == "__main__":
    main()
