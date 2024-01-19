import numpy as np
import matplotlib.pyplot as plt
import state


def target_distribution(x):
    # Define the target distribution, for example, a Gaussian distribution
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def metropolis_algorithm(initial_state, num_samples, proposal_std):
    samples = [initial_state]
    current_state = initial_state

    for _ in range(num_samples):
        # Propose a new state from a normal distribution
        proposed_state = np.random.normal(current_state, proposal_std)

        # Calculate the acceptance ratio
        acceptance_ratio = min(
            1, target_distribution(proposed_state) / target_distribution(current_state)
        )

        # Accept or reject the proposed state
        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state

        samples.append(current_state)

    return np.array(samples)


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
    # test2 = state.SquareSystemNonPeriodicState(3)

    # test.get_state()[1] = 1  # direct manipulation IS possible
    # print(test.get_state())
    # test2.get_state()[1] = 1

    # print(test.scalar_product(test2))

    # print(test.get_nearest_neighbor_indices(4))
    # print(test.get_nearest_neighbor_indices(7))
    # print(test.get_nearest_neighbor_indices(13))
    # print(test.get_nearest_neighbor_indices(16))
    # print(test.get_nearest_neighbor_indices(17))
