import multiprocessing


def worker_function(task):
    # This is the function that each worker process will execute
    # You can put your task processing logic here
    result = task * task

    while True:
        task * task

    return result


def main():
    # Define your tasks
    tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Create a multiprocessing Pool with the number of CPUs you want to utilize
    num_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cpus)

    # Distribute the tasks across the pool of worker processes
    results = pool.map(worker_function, tasks)

    # Close the pool and wait for all tasks to complete
    pool.close()
    pool.join()

    # Print the results
    print("Results:", results)


if __name__ == "__main__":
    main()
