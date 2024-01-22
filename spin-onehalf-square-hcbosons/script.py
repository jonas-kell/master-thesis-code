import numpy as np

import state
import sampler
import hamiltonian
from randomgenerator import RandomGenerator


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # # Parameters
    # initial_state = 0.0
    # num_samples = 100000
    # proposal_std = 0.5
    # # Run Metropolis algorithm
    # samples = metropolis_algorithm(initial_state, num_samples, proposal_std)
    # # Plot the results
    # plt.hist(samples, bins=50, density=True, label="Samples")
    # x_range = np.linspace(-3, 3, 100)
    # plt.plot(
    #     x_range, target_distribution(x_range), label="Target Distribution", color="red"
    # )
    # plt.title("Metropolis Algorithm for Monte Carlo Sampling")
    # plt.legend()
    # plt.show()

    test = state.SquareSystemNonPeriodicState(3)

    # test.get_state_array()[1] = 1  # direct manipulation IS possible
    # print(test.get_state_array())
    # print(test.get_nearest_neighbor_indices(4))
    # print(test.get_nearest_neighbor_indices(7))
    # print(test.get_nearest_neighbor_indices(13))
    # print(test.get_nearest_neighbor_indices(16))
    # print(test.get_nearest_neighbor_indices(17))

    generator = RandomGenerator("testabqasdsa")

    # print(test.get_state_array())
    test.init_random_filling(0.5, generator)
    print(test.get_state_array())

    U = 0.4
    E = 0.4
    J = 0.4
    phi = np.pi / 4
    ham = hamiltonian.HardcoreBosonicHamiltonian(U=U, E=E, J=J, phi=phi)

    beta = 0.4
    state_sampler = sampler.MonteCarloSampler(
        system_state=test,
        beta=beta,
        system_hamiltonian=ham,
        generator=generator,
        no_intermediate_mc_steps=100,
        no_random_swaps=2,
        no_samples=20000,  # 262144
        no_thermalization_steps=10000,
    )

    # state_sampler = sampler.ExactSampler(
    #     system_state=test,
    # )

    sample_generator_object = state_sampler.sample_generator()

    sample_count = 0
    total_sum = np.zeros((1,), dtype=np.complex128)[0]
    time = 0
    while True:
        try:
            sampled_state_n = next(sample_generator_object)
            ## generate averages using sampled state
            sample_count += 1

            h_eff = ham.get_exp_H_effective_of_n_and_t(
                time=time,
                system_state_object=sampled_state_n,
                system_state_array=sampled_state_n.get_state_array(),
            )
            psi_n = sampled_state_n.get_Psi_of_N(sampled_state_n.get_state_array())
            total_sum += np.real(
                np.conj(h_eff) * h_eff  # keep the autoformatter from rewriting ugly...
            ) * np.real(
                np.conj(psi_n) * psi_n  #
            )

            if sample_count % 1000 == 0:
                print(sample_count)

            ## end generate averages using sampled state
        except StopIteration:
            break

    print(total_sum)
    print(sample_count)
