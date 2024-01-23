import hamiltonian as hamiltonianImport
import observables as observablesImport
import sampler as samplerImport
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


def main_measurement_function(
    state_sampler: samplerImport.GeneralSampler,
    hamiltonian: hamiltonianImport.Hamiltonian,
    observable: observablesImport.Observable,
    start_time: float,
    time_step: float,
    number_of_time_steps: int,
) -> Tuple[List[float], List[float]]:
    time_list: List(float) = []
    values_list: List(float) = []

    for time_step_nr in range(number_of_time_steps):
        time: float = start_time + time_step * time_step_nr

        sample_generator_object = state_sampler.sample_generator()

        sample_count = 0
        total_sum: float = 0.0
        while True:
            try:
                sampled_state_n = next(sample_generator_object)
                ## generate averages using sampled state
                sample_count += 1

                h_eff = hamiltonian.get_exp_H_effective_of_n_and_t(
                    time=time,
                    system_state_object=sampled_state_n,
                )
                psi_n = sampled_state_n.get_Psi_of_N(sampled_state_n.get_state_array())
                observed_quantity = observable.get_expectation_value(sampled_state_n)

                total_sum += (
                    np.real(np.conj(h_eff) * h_eff)
                    * np.real(np.conj(psi_n) * psi_n)
                    * observed_quantity
                )

                # if sample_count % 1000 == 0:
                #     print(sample_count)

                ## end generate averages using sampled state
            except StopIteration:
                break

        sampled_value: float = total_sum

        print(f"Time: {time:.3f} {sampled_value} ({sample_count} samples)")
        time_list.append(time)
        values_list.append(sampled_value)

    return (time_list, values_list)


def plot_measurements(
    times: List[float], values: List[float], title: str, x_label: str, y_label: str
):
    # Plot the results
    plt.plot(times, values, color="red")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
