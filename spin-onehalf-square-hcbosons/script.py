import numpy as np
import matplotlib.pyplot as plt
import state
import montecarlo
import hamiltonian
from randomgenerator import RandomGenerator

if __name__ == "__main__":
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

    generator = RandomGenerator("testabq")

    print(test.get_state_array())
    print(test.init_random_filling(0.5, generator))
    print(test.get_state_array())

    beta = 0.4
    U = 0.4
    E = 0.4
    J = 0.4
    phi = np.pi / 4
    ham = hamiltonian.HardcoreBosonicHamiltonian(U=U, E=E, J=J, phi=phi)

    sampler = montecarlo.MonteCarloSampler(
        system_state=test,
        beta=beta,
        system_hamiltonian=ham,
        generator=generator,
    )

    sampler.do_metropolis_steps(100)
    print(test.get_state_array())
