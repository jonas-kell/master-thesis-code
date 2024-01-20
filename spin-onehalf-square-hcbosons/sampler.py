import numpy as np
import state
from randomgenerator import RandomGenerator
import hamiltonian
from abc import ABC, abstractmethod
from typing import Generator


class GeneralSampler(ABC):
    def __init__(
        self,
        system_state: state.SystemState,
    ):
        self.system_state = system_state

    @abstractmethod
    def sample_generator(self) -> Generator[state.SystemState, None, None]:
        """
        Generator of system states. Call next(gen) to get more SystemStates
        """
        pass


class MonteCarloSampler(GeneralSampler):
    """
    beta is basically 1/(k_B*T) and should have the same scale as the energy function
    """

    def __init__(
        self,
        system_state: state.SystemState,
        beta: float,
        system_hamiltonian: hamiltonian.Hamiltonian,
        generator: RandomGenerator,
        no_samples: int,
        no_intermediate_mc_steps: int,
        no_random_swaps: int,
        no_thermalization_steps: int,
    ):
        self.beta = beta
        self.system_hamiltonian = system_hamiltonian
        self.generator = generator
        self.no_samples = no_samples
        self.no_intermediate_mc_steps = no_intermediate_mc_steps
        self.no_random_swaps = no_random_swaps
        self.no_thermalization_steps = no_thermalization_steps
        self.thermalized = False
        super().__init__(system_state=system_state)

    def do_metropolis_steps(self, num_steps: int) -> None:
        """
        Advances the system_state in-place by num_steps metropolis steps
        """
        for _ in range(num_steps):
            # Propose a new state by random swap
            original_state_array = self.system_state.get_state_array()
            proposed_state_array = self.system_state.get_random_swap_copy(
                no_swaps=self.no_random_swaps, generator=self.generator
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

    def thermalize(self):
        if not self.thermalized:
            self.do_metropolis_steps(self.no_thermalization_steps)
            self.thermalized = True

    def sample_generator(self) -> Generator[state.SystemState, None, None]:
        self.thermalize()  # make sure, to thermalize the system at least once

        for _ in range(self.no_samples):
            yield self.system_state
            self.do_metropolis_steps(self.no_intermediate_mc_steps)


class ExactSampler(GeneralSampler):
    """
    beta is basically 1/(k_B*T) and should have the same scale as the energy function
    """

    def __init__(
        self,
        system_state: state.SystemState,
    ):
        super().__init__(system_state=system_state)

    def sample_generator(self) -> Generator[state.SystemState, None, None]:
        working_state = np.zeros_like(self.system_state.get_state_array())
        self.system_state.set_state(working_state)

        number_samples = 2 ** self.system_state.get_number_sites()

        array_length = working_state.shape[0]

        for _ in range(number_samples):
            yield self.system_state

            carry = 1
            for i in range(array_length):
                res = working_state[i] + carry

                working_state[i] = res % 2
                carry = res // 2
