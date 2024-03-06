import state
from randomgenerator import RandomGenerator
import hamiltonian
from abc import ABC, abstractmethod
from typing import Generator
import math


class GeneralSampler(ABC):
    def __init__(
        self,
        system_geometry: state.SystemGeometry,
        initial_system_state: state.InitialSystemState,
    ):
        self.system_geometry = system_geometry
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
        number_samples = 2 ** self.system_geometry.get_number_sites()
        return number_samples

    @abstractmethod
    def produces_samples_count(self) -> int:
        pass


class MonteCarloSampler(GeneralSampler):
    def __init__(
        self,
        system_geometry: state.SystemGeometry,
        initial_system_state: state.InitialSystemState,
        system_hamiltonian: hamiltonian.Hamiltonian,
        random_generator: RandomGenerator,
        num_samples: int,
        num_intermediate_mc_steps: int,
        num_random_flips: int,
        num_thermalization_steps: int,
        initial_fill_level: float,
    ):
        super().__init__(
            system_geometry=system_geometry, initial_system_state=initial_system_state
        )
        self.system_hamiltonian = system_hamiltonian
        self.random_generator = random_generator
        self.num_samples = num_samples
        self.num_intermediate_mc_steps = num_intermediate_mc_steps
        self.num_random_flips = num_random_flips
        self.num_thermalization_steps = num_thermalization_steps
        self.thermalized = False
        self.initial_fill_level = initial_fill_level

    def do_metropolis_steps(
        self, state_to_modify: state.SystemState, num_steps: int, time: float
    ) -> None:
        """
        Advances the system_state in-place by num_steps metropolis steps
        """
        for _ in range(num_steps):
            # Propose a new state by random swap
            original_state = state_to_modify
            proposed_state = state_to_modify.get_random_flipped_copy(
                num_flips=self.num_random_flips, random_generator=self.random_generator
            )

            # Calculate the energies and probabilities
            original_state_psi = original_state.get_Psi_of_N()
            original_state_energy_exp = (
                self.system_hamiltonian.get_exp_H_effective_of_n_and_t(
                    system_state=original_state,
                    time=time,
                )
            )
            proposed_state_psi = proposed_state.get_Psi_of_N()
            proposed_state_energy_exp = (
                self.system_hamiltonian.get_exp_H_effective_of_n_and_t(
                    system_state=proposed_state,
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
                state_to_modify.set_state_array(proposed_state.get_state_array())

    def initialize_fill_level(self, state_to_modify: state.SystemState):
        state_to_modify.init_random_filling(
            fill_ratio=self.initial_fill_level,
            random_generator=self.random_generator,
        )
        self.thermalized = False

    def thermalize(self, state_to_modify: state.SystemState, time: float):
        self.do_metropolis_steps(
            state_to_modify, num_steps=self.num_thermalization_steps, time=time
        )

    def sample_generator(
        self, time: float, worker_index: int, num_workers: int
    ) -> Generator[state.SystemState, None, None]:
        _ = worker_index  # this generator is independent of worker_index

        working_state = state.SystemState(
            system_geometry=self.system_geometry,
            initial_system_state=self.initial_system_state,
        )

        self.initialize_fill_level(working_state)
        self.thermalize(working_state, time)

        # TODO Support multiple chain-runs per worker

        for _ in range(math.ceil(self.num_samples / num_workers)):
            yield working_state
            self.do_metropolis_steps(
                working_state, num_steps=self.num_intermediate_mc_steps, time=time
            )

    def produces_samples_count(self) -> int:
        return self.num_samples


class ExactSampler(GeneralSampler):
    def __init__(
        self,
        system_geometry: state.SystemGeometry,
        initial_system_state: state.InitialSystemState,
    ):
        super().__init__(
            system_geometry=system_geometry, initial_system_state=initial_system_state
        )

    def sample_generator(
        self, time: float, worker_index: int, num_workers: int
    ) -> Generator[state.SystemState, None, None]:
        _ = time  # this generator is independent of time

        working_state = state.SystemState(
            system_geometry=self.system_geometry,
            initial_system_state=self.initial_system_state,
        )
        array_length = working_state.get_state_array().shape[0]

        # TODO init state and upper bound
        _ = worker_index
        _ = num_workers

        for _ in range(self.all_samples_count()):
            yield working_state

            carry = 1
            for i in range(array_length):
                res = working_state.get_state_array()[i] + carry

                working_state.get_state_array()[i] = res % 2
                carry = res // 2

    def produces_samples_count(self) -> int:
        return self.all_samples_count()
