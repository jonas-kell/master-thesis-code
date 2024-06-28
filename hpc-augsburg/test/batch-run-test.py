import os
import multiprocessing


def main():
    env_var_name = "SLURM_ARRAY_TASK_ID"
    env_value = os.getenv(env_var_name)

    if env_value is None:
        print(f"Environment variable {env_var_name} is not set.")
        return

    try:
        # Parse the value to an integer
        int_value = int(env_value)
    except ValueError:
        print(f"Cannot convert {env_value} to an integer.")
        return

    # Define the filename
    filename = f"result-{int_value}.txt"

    try:
        # Write the integer value to the file
        with open(filename, "w") as file:
            file.write(str(int_value))
        print(f"Successfully wrote to {filename}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

    print(f"Available Core Count: {multiprocessing.cpu_count()}")


if __name__ == "__main__":
    main()
