import numpy as np
import state
from typing import Callable
from randomgenerator import RandomGenerator
import hamiltonian


class MonteCarloSampler:
    """
    beta is basically 1/(k_B*T) and should have the same scale as the energy function
    """

    def __init__(
        self,
        system_state: state.SystemState,
        beta: float,
        system_hamiltonian: hamiltonian.Hamiltonian,
        generator: RandomGenerator,
    ):
        self.system_state = system_state
        self.beta = beta
        self.system_hamiltonian = system_hamiltonian
        self.generator = generator

    def do_metropolis_steps(self, num_steps: int) -> None:
        """
        Advances the system_state in-place by num_steps metropolis steps
        """
        for _ in range(num_steps):
            # Propose a new state by random swap
            original_state_array = self.system_state.get_state_array()
            proposed_state_array = self.system_state.get_random_swap_copy(
                generator=self.generator
            )

            # Calculate the energies
            original_state_energy = self.system_hamiltonian.get_base_energy(
                self.system_state, original_state_array
            )

            proposed_state_energy = self.system_hamiltonian.get_base_energy(
                self.system_state, proposed_state_array
            )

            # Calculate the acceptance ratio
            #
            # if proposed_state_energy (final) > original_state_energy (initial), then
            # exp (-positive) < 1 -> exp is taken
            #
            # if proposed_state_energy (final) <= original_state_energy (initial), then
            # exp (-negative) > 1 -> 1 is taken
            acceptance_ratio = min(
                1, np.exp(-self.beta * (proposed_state_energy - original_state_energy))
            )

            # Accept or reject the proposed state
            if self.generator.probability() < acceptance_ratio:
                self.system_state.set_state(proposed_state_array)

    def target_distribution(x):
        # Define the target distribution, for example, a Gaussian distribution
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
