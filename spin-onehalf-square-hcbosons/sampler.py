import numpy as np
import state
from randomgenerator import RandomGenerator
import hamiltonian
from abc import ABC, abstractmethod
from typing import Generator
import math


class GeneralSampler(ABC):
    def __init__(
        self,
        system_state: state.SystemState,
        initial_system_state: state.InitialSystemState,
    ):
        self.system_state = system_state
        self.initial_system_state = initial_system_state

    @abstractmethod
    def sample_generator(
        self, time: float, worker_index: int, num_workers: int
    ) -> Generator[state.SystemState, None, None]:
        """
        Generator of system states. Call next(gen) to get more SystemStates
        """
        pass

    def all_samples_count(self) -> int:
        number_samples = 2 ** self.system_state.get_number_sites()

        return number_samples

    @abstractmethod
    def produces_samples_count(self) -> int:
        pass


class MonteCarloSampler(GeneralSampler):
    def __init__(
        self,
        system_state: state.SystemState,
        initial_system_state: state.InitialSystemState,
        system_hamiltonian: hamiltonian.Hamiltonian,
        random_generator: RandomGenerator,
        num_samples: int,
        num_intermediate_mc_steps: int,
        num_random_flips: int,
        num_thermalization_steps: int,
        initial_fill_level: float,
    ):
        self.system_hamiltonian = system_hamiltonian
        self.random_generator = random_generator
        self.num_samples = num_samples
        self.num_intermediate_mc_steps = num_intermediate_mc_steps
        self.num_random_flips = num_random_flips
        self.num_thermalization_steps = num_thermalization_steps
        self.thermalized = False
        self.initial_fill_level = initial_fill_level
        super().__init__(
            system_state=system_state, initial_system_state=initial_system_state
        )

    def do_metropolis_steps(self, num_steps: int, time: float) -> None:
        """
        Advances the system_state in-place by num_steps metropolis steps
        """
        for _ in range(num_steps):
            # Propose a new state by random swap
            original_state_array = self.system_state.get_state_array()
            proposed_state_array = self.system_state.get_random_flipped_copy(
                num_flips=self.num_random_flips, random_generator=self.random_generator
            )

            # Calculate the energies and probabilities
            original_state_psi = self.initial_system_state.get_Psi_of_N(
                system_state_array=original_state_array
            )
            original_state_energy_exp = (
                self.system_hamiltonian.get_exp_H_effective_of_n_and_t(
                    system_state_object=self.system_state,
                    system_state_array=original_state_array,
                    initial_system_state=self.initial_system_state,
                    time=time,
                )
            )
            proposed_state_psi = self.initial_system_state.get_Psi_of_N(
                system_state_array=proposed_state_array
            )
            proposed_state_energy_exp = (
                self.system_hamiltonian.get_exp_H_effective_of_n_and_t(
                    system_state_object=self.system_state,
                    initial_system_state=self.initial_system_state,
                    system_state_array=proposed_state_array,
                    time=time,
                )
            )

            # Calculate the acceptance ratio
            #
            # if proposed_state_energy (final) > original_state_energy (initial), then
            # exp (-positive) < 1 -> exp is taken
            #
            # if proposed_state_energy (final) <= original_state_energy (initial), then
            # exp (-negative) > 1 -> 1 is taken
            acceptance_ratio = min(
                1,
                (
                    (proposed_state_psi * proposed_state_energy_exp)
                    / (original_state_psi * original_state_energy_exp)
                ),
            )

            # Accept or reject the proposed state
            if self.random_generator.probability() < acceptance_ratio:
                self.system_state.set_state(proposed_state_array)

    def initialize_fill_level(self):
        self.system_state.init_random_filling(
            fill_ratio=self.initial_fill_level,
            random_generator=self.random_generator,
        )
        self.thermalized = False

    def thermalize(self, time: float):
        if not self.thermalized:
            self.do_metropolis_steps(self.num_thermalization_steps, time=time)
            self.thermalized = True

    def sample_generator(
        self, time: float, worker_index: int, num_workers: int
    ) -> Generator[state.SystemState, None, None]:
        worker_index  # this generator is independent of worker_index

        self.initialize_fill_level()
        self.thermalize(time)  # make sure, to thermalize the system at least once

        # TODO Support multiple chain-runs per worker

        # TODO rewrite so that system_state object is not shared, as this breaks if multiple processes use one reference

        for _ in range(math.ceil(self.num_samples / num_workers)):
            yield self.system_state
            self.do_metropolis_steps(self.num_intermediate_mc_steps, time=time)

    def produces_samples_count(self) -> int:
        return self.num_samples


class ExactSampler(GeneralSampler):
    def __init__(
        self,
        system_state: state.SystemState,
        initial_system_state: state.InitialSystemState,
    ):
        super().__init__(
            system_state=system_state, initial_system_state=initial_system_state
        )

    def sample_generator(
        self, time: float, worker_index: int, num_workers: int
    ) -> Generator[state.SystemState, None, None]:
        time  # this generator is independent of time

        working_state = np.zeros_like(self.system_state.get_state_array())
        array_length = working_state.shape[0]
        self.system_state.set_state(working_state)

        # TODO rewrite so that system_state object is not shared, as this breaks if multiple processes use one reference

        for _ in range(self.all_samples_count()):
            yield self.system_state

            carry = 1
            for i in range(array_length):
                res = working_state[i] + carry

                working_state[i] = res % 2
                carry = res // 2

    def produces_samples_count(self) -> int:
        return self.all_samples_count()
