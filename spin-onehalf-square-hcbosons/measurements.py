import hamiltonian as hamiltonianImport
import observables as observablesImport
import sampler as samplerImport
import state as stateImport
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import time as computerTime


def main_measurement_function(
    state_sampler: samplerImport.GeneralSampler,
    hamiltonian: hamiltonianImport.Hamiltonian,
    observables: List[observablesImport.Observable],
    initial_system_state: stateImport.InitialSystemState,
    start_time: float,
    time_step: float,
    number_of_time_steps: int,
    plot: bool = False,
    plot_title: str = "Calculations on Spin System",
) -> Tuple[List[float], List[List[float]]]:
    time_list: List[float] = []
    values_list: List[List[float]] = []

    exact_sample_count = state_sampler.all_samples_count()
    used_sample_count = state_sampler.produces_samples_count()
    correction_fraction = float(exact_sample_count) / used_sample_count

    function_start_time = None
    sample_count = 0
    total_needed_sample_count = used_sample_count * number_of_time_steps

    # DEFAULT PRINTS
    default_prints = True

    no_observables = len(observables)

    for time_step_nr in range(number_of_time_steps):
        step_sample_count = 0
        time: float = start_time + time_step * time_step_nr

        sample_generator_object = state_sampler.sample_generator(time=time)

        total_sums: List[float] = [0.0] * no_observables
        while True:
            try:
                # track time from start (includes thermalization, because will be required when we use multiple cores/generator-chains for processing)
                if function_start_time is None:
                    function_start_time = computerTime.time()
                sampled_state_n = next(sample_generator_object)

                ## generate averages using sampled state
                sample_count += 1
                step_sample_count += 1

                h_eff = hamiltonian.get_exp_H_effective_of_n_and_t(
                    time=time,
                    system_state_object=sampled_state_n,
                    initial_system_state=initial_system_state,
                    system_state_array=sampled_state_n.get_state_array(),
                )
                psi_n = initial_system_state.get_Psi_of_N(
                    sampled_state_n.get_state_array()
                )

                energy_factor: float = np.real(np.conj(h_eff) * h_eff) * np.real(
                    np.conj(psi_n) * psi_n
                )

                for i, observable in enumerate(observables):
                    observed_quantity = observable.get_expectation_value(
                        sampled_state_n
                    )
                    total_sums[i] += energy_factor * observed_quantity

                if sample_count % 1000 == 0:
                    percentage = sample_count / total_needed_sample_count * 100
                    time_needed_so_far = computerTime.time() - function_start_time
                    probable_total_time = time_needed_so_far / percentage * 100

                    if default_prints:
                        print(
                            f"In total sampled {sample_count} of {total_needed_sample_count}. Took {time_needed_so_far:.2f}s ({percentage:.1f}%). 100% prognosis: {probable_total_time:.1f}s ({probable_total_time-time_needed_so_far:.1f}s remaining)"
                        )

                ## end generate averages using sampled state
            except StopIteration:
                break

        # scale observables
        for i in range(no_observables):
            total_sums[i] *= correction_fraction

        if default_prints:
            print(
                f"Time: {time:.3f} {total_sums[0]:2.5f} ({step_sample_count} samples, while exact needs {exact_sample_count})"
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
            observables=observables,
            title=plot_title,
            x_label="time t",
        )

    return (time_list, values_list)


def plot_measurements(
    times: List[float],
    values: List[List[float]],
    observables: List[observablesImport.Observable],
    title: str,
    x_label: str,
):
    data = np.array(values).T  # Transpose to extract form we want to plot

    num_observables = len(observables)
    num_rows = max(
        1, int(np.ceil(num_observables / np.sqrt(num_observables)))
    )  # Divide by root to get approximate square arrangement
    num_cols = int(np.ceil(num_observables / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 6))

    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    else:
        if num_rows == 1 or num_cols == 1:
            axes = np.array([axes]).T

    for i, obs in enumerate(observables):
        row = i // num_cols
        col = i % num_cols

        # Plot the results
        axes[row, col].plot(times, data[i], color="red")
        axes[row, col].set_xlabel(x_label)
        axes[row, col].set_ylabel(obs.get_label())

    # Remove any unused subplots
    for i in range(num_observables, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
