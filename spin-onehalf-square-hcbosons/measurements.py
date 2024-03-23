import hamiltonian as hamiltonianImport
import observables as observablesImport
import sampler as samplerImport
import numpy as np
import matplotlib.pyplot as plt
import time as computerTime
import multiprocessing
from datetime import datetime
import os
import json
from typing import Dict, Union, Any, Tuple, List, cast
from randomgenerator import RandomGenerator


def main_measurement_function(
    state_sampler: samplerImport.GeneralSampler,
    random_generator: RandomGenerator,
    hamiltonian: hamiltonianImport.Hamiltonian,
    observables: List[observablesImport.Observable],
    start_time: float,
    time_step: float,
    number_of_time_steps: int,
    number_workers: int,
    plot: bool = False,
    plot_title: str = "Calculations on Spin System",
    plot_x_label: str = "time t",
    write_to_file: bool = True,
) -> Tuple[List[float], List[List[float]]]:
    time_list: List[float] = []
    values_list: List[List[float]] = []

    exact_sample_count = state_sampler.all_samples_count()
    used_sample_count = state_sampler.produces_samples_count()

    num_observables = len(observables)

    function_start_time = computerTime.time()
    total_needed_sample_count = used_sample_count * number_of_time_steps
    total_sample_count = 0

    # file-writing
    current_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./../run-outputs/" + datetime.now().strftime("%Y-%m-%d__%H,%M,%S") + ".json",
    )
    if write_to_file:  # write the base file information
        with open(current_file_path, mode="w", newline="") as file:
            json.dump(
                {
                    "hamiltonian": hamiltonian.get_log_info(),
                    "start_time": start_time,
                    "time_step": time_step,
                    "number_of_time_steps": number_of_time_steps,
                    "sampler": state_sampler.get_log_info(),
                    "observables": [
                        observable.get_log_info() for observable in observables
                    ],
                    "plot_title": plot_title,
                    "plot_x_label": plot_x_label,
                    "measurements": [],
                    "random_generator": random_generator.get_log_info(),
                },
                file,
            )

    # DEFAULT PRINTS
    default_prints = True
    if default_prints:
        print(f"Started measurement process with {number_workers} workers")

    for time_step_nr in range(number_of_time_steps):
        step_sample_count = 0
        time: float = start_time + time_step * time_step_nr

        total_sums_complex: List[np.complex128] = [np.complex128(0.0)] * num_observables

        # ! Branch out jobs into worker-functions

        pool = multiprocessing.Pool(processes=number_workers)

        tasks = [
            (
                job_number,
                time,
                observables,
                state_sampler,
                hamiltonian,
                random_generator.derive(),
                number_workers,
                total_sample_count,
                total_needed_sample_count,
                function_start_time,
                # only one core reports approximate status to save on necessary joining/waiting/coordinating
                default_prints and job_number == 0,
            )
            for job_number in range(number_workers)
        ]

        results = pool.starmap(run_worker_chain, tasks)
        pool.close()
        pool.join()

        normalization_factor: float = 0

        # ! collect results from threads
        for result in results:
            worker_sample_count, worker_normalization_factor, worker_sums = result

            step_sample_count += worker_sample_count
            total_sample_count += worker_sample_count
            normalization_factor += worker_normalization_factor
            for i in range(num_observables):
                total_sums_complex[i] += worker_sums[i]
        # ! Collected branched out jobs from worker-functions

        inverse_normalization_factor = 1 / normalization_factor

        # scale and convert observables
        total_sums: List[float] = [0.0] * num_observables
        for i in range(num_observables):
            imag_part_of_observable = float(
                np.imag(total_sums_complex[i]) * inverse_normalization_factor
            )
            if np.abs(imag_part_of_observable) < 1e-6:
                print(
                    f"Warning observable had imaginary part of {imag_part_of_observable:.5f} that was omitted"
                )
            total_sums[i] = float(
                np.real(total_sums_complex[i]) * inverse_normalization_factor
            )

        if default_prints:
            print(
                f"Time: {time:.3f} (step {time_step_nr+1}/{number_of_time_steps}) {total_sums[0]:2.5f} ({step_sample_count} samples, while exact needs {exact_sample_count})"
            )
        if write_to_file:  # write the base file information
            data: Dict[str, Union[float, str, Dict[Any, Any], List[Any]]] = {}
            with open(current_file_path, mode="r") as file:
                data = json.load(file)
            with open(current_file_path, mode="w", newline="") as file:
                measurements: List[Dict[Any, Any]] = data["measurements"]  # type: ignore

                measurements.append(
                    {
                        "time": time,
                        "step_sample_count": step_sample_count,
                        "data": total_sums,
                    }
                )

                data["measurements"] = measurements

                json.dump(
                    data,
                    file,
                )

        time_list.append(time)
        values_list.append(total_sums)

    if default_prints:
        print(
            f"Whole computation took {computerTime.time()-function_start_time:.3f} seconds"
        )

    if plot:
        plot_measurements(
            times=time_list,
            values=values_list,
            observable_labels=[obs.get_label() for obs in observables],
            title=plot_title,
            x_label=plot_x_label,
        )

    return (time_list, values_list)


def run_worker_chain(
    job_number: int,
    time: float,
    observables: List[observablesImport.Observable],
    state_sampler: samplerImport.GeneralSampler,
    hamiltonian: hamiltonianImport.Hamiltonian,
    random_generator: RandomGenerator,
    number_workers: int,
    total_sample_count: int,
    total_needed_sample_count: int,
    function_start_time: float,
    default_prints: bool,
) -> Tuple[int, float, List[np.complex128]]:
    """
    returns: (worker_sample_count, worker_sums)
    """
    num_observables = len(observables)
    worker_sums: List[np.complex128] = [np.complex128(0.0)] * num_observables

    worker_sample_count = 0
    normalization_factor = 0.0

    sample_generator_object = state_sampler.sample_generator(
        time=time,
        num_workers=number_workers,
        worker_index=job_number,
        random_generator=random_generator,
    )

    # report at the start of an iteration definitely
    last_worker_report_time = computerTime.time() - 10.1

    requires_probability_adjustment = state_sampler.requires_probability_adjustment()

    while True:
        try:
            sampled_state_n = next(sample_generator_object)

            ## generate measurements using sampled state
            worker_sample_count += 1

            if requires_probability_adjustment:
                # sampled state needs to be scaled

                # get_exp_H_eff is the most expensive calculation. Only do if absolutely necessary
                h_eff = hamiltonian.get_exp_H_eff(
                    time=time, system_state=sampled_state_n
                )
                psi_n = sampled_state_n.get_Psi_of_N()

                state_probability: float = np.real(np.conjugate(h_eff) * h_eff) * np.real(  # type: ignore -> this returns a scalar for sure
                    np.conjugate(psi_n) * psi_n
                )
            else:
                # e.g. Monte Carlo. Normalization is only division by number of monte carlo samples
                state_probability = 1.0

            normalization_factor += state_probability

            for i, observable in enumerate(observables):
                observed_quantity = observable.get_expectation_value(
                    time=time, system_state=sampled_state_n
                )
                worker_sums[i] += state_probability * observed_quantity

            ## end generate measurements using sampled state

            if worker_sample_count % 10 == 0:
                current_time = computerTime.time()
                if current_time - last_worker_report_time > 10:
                    last_worker_report_time = current_time

                    extrapolated_sample_count = (
                        total_sample_count + worker_sample_count * number_workers
                    )
                    percentage = (
                        extrapolated_sample_count / total_needed_sample_count * 100
                    )
                    time_needed_so_far = current_time - function_start_time
                    probable_total_time = time_needed_so_far / percentage * 100

                    if default_prints:
                        print(
                            f"In total sampled (extrapolated) {extrapolated_sample_count} of {total_needed_sample_count}. Took {time_needed_so_far:.2f}s ({percentage:.1f}%). 100% prognosis: {probable_total_time:.1f}s ({probable_total_time-time_needed_so_far:.1f}s remaining)"
                        )

        except StopIteration:
            break

    return worker_sample_count, normalization_factor, worker_sums


def plot_measurements(
    times: List[float],
    values: List[List[float]],
    observable_labels: List[str],
    title: str,
    x_label: str,
):
    data = np.array(values).T  # Transpose to extract form we want to plot

    num_observables = len(observable_labels)
    num_rows = max(
        1, int(np.ceil(num_observables / np.sqrt(num_observables)))
    )  # Divide by root to get approximate square arrangement
    num_cols = int(np.ceil(num_observables / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 6))  # type: ignore -> matplotlib typing is non-existent
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])  # type: ignore -> matplotlib typing is non-existent
    else:
        if num_rows == 1 or num_cols == 1:
            axes = np.array([axes]).T  # type: ignore -> matplotlib typing is non-existent

    for i, obs in enumerate(observable_labels):
        row = i // num_cols
        col = i % num_cols

        # Plot the results
        axes[row, col].plot(times, data[i], color="red")  # type: ignore -> matplotlib typing is non-existent
        axes[row, col].set_xlabel(x_label)  # type: ignore -> matplotlib typing is non-existent
        axes[row, col].set_ylabel(obs)  # type: ignore -> matplotlib typing is non-existent

    # Remove any unused subplots
    for i in range(num_observables, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    plt.suptitle(title)  # type: ignore -> matplotlib typing is non-existent
    plt.tight_layout()  # type: ignore -> matplotlib typing is non-existent
    plt.show()  # type: ignore -> matplotlib typing is non-existent


if __name__ == "__main__":
    file_name = "2024-XX-XX__XX,XX,XX.json"
    current_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./../run-outputs/" + file_name,
    )

    print(f"plotting measurements from file: {current_file_path}")

    data: Dict[str, Union[float, str, Dict[Any, Any], List[Any]]] = {}
    with open(current_file_path, mode="r") as file:
        data = json.load(file)

    title: str = data["plot_title"]  # type: ignore
    plot_x_label: str = data["plot_x_label"]  # type: ignore

    times: List[float] = []
    values: List[List[float]] = []
    for tmp in data["measurements"]:  # type: ignore
        time = cast(float, tmp["time"])
        value = cast(List[float], tmp["data"])

        times.append(time)
        values.append(value)

    observable_labels: List[str] = [tmp["label"] for tmp in data["observables"]]  # type: ignore

    plot_measurements(
        observable_labels=observable_labels,
        times=times,
        values=values,
        title=title,
        x_label=plot_x_label,
    )
