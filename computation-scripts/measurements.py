from typing import Dict, Union, Any, Tuple, List
import time as computerTime
import multiprocessing
from datetime import datetime
import os
import json
import hamiltonian as hamiltonianImport
import observables as observablesImport
import sampler as samplerImport
import numpy as np
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
    job_array_index: int,
    write_to_file: bool = True,
    file_name_overwrite: str | None = None,
    check_obs_imag: bool = False,
    check_obs_imag_threshold: float = 1e-2,
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
    time_canonical_name = (
        datetime.now().strftime("%Y-%m-%d__%H,%M,%S") + "-" + str(job_array_index)
    )
    if file_name_overwrite is not None:
        # overwrite the file's name
        time_canonical_name = file_name_overwrite
    current_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./../run-outputs/" + time_canonical_name + ".json",
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

        # e.g. the VCN hamiltonian requires initializing (training )
        hamiltonian.initialize(time=time)

        total_sums_complex: List[np.complex128 | np.ndarray] = [
            np.complex128(0.0)
        ] * num_observables

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
        for i, observable in enumerate(observables):
            measurement = total_sums_complex[i] * inverse_normalization_factor

            if observable.post_process_necessary():
                # in this case, the handled values in an ndarray that needs post processing (e.g. a density matrix or other factors that need to be sampled)
                measurement = observable.post_process(measurement)

            real_part_of_observable = float(np.real(measurement))
            imag_part_of_observable = float(np.imag(measurement))
            if np.abs(imag_part_of_observable) > check_obs_imag_threshold:
                message = f"Observable {observable.get_label()} with real part {real_part_of_observable:.6f} had imaginary part of {imag_part_of_observable:.6f} that was omitted"
                if check_obs_imag:
                    raise Exception(message)
                else:
                    print("Warning:", message)
            total_sums[i] = real_part_of_observable

        if default_prints:
            print(
                f"Time: {time:.3f} (step {time_step_nr+1}/{number_of_time_steps}) {total_sums} ({step_sample_count} samples, while exact needs {exact_sample_count})"
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
                data["realworld_total_calculation_time"] = float(
                    f"{computerTime.time()-function_start_time:.3f}"
                )

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
) -> Tuple[int, float, List[np.complex128 | np.ndarray]]:
    """
    returns: (worker_sample_count, worker_sums)
    """
    num_observables = len(observables)
    # when one observable outputs a np.ndarray in get_expectation_value(), this will be correctly handled
    worker_sums: List[np.complex128 | np.ndarray] = [
        np.complex128(0.0)
    ] * num_observables

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

    #! extract probabilities during run #TODO remove
    # probs = []

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

            #! extract probabilities during run #TODO remove
            # probs.append(
            #     (np.copy(sampled_state_n.get_state_array()), state_probability)
            # )

            for i, observable in enumerate(observables):
                observed_quantity = observable.get_expectation_value(
                    time=time, system_state=sampled_state_n
                )
                # this operation is correctly handled if the return type is an array (thanks python for once)
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

    #! extract probabilities during run #TODO remove
    # if time > 390:
    #     print("perturbation")
    #     for arr, prob in probs:
    #         print(arr)
    #         print(prob / normalization_factor)
    #     exit()

    return worker_sample_count, normalization_factor, worker_sums
