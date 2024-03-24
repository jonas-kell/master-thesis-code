import state
from randomgenerator import RandomGenerator
import hamiltonian
from abc import ABC, abstractmethod
from typing import Generator
import math
from typing import Dict, Union, Any
import numpy as np


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
        self,
        time: float,
        worker_index: int,
        num_workers: int,
        random_generator: Union[RandomGenerator, None],
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

    @abstractmethod
    def requires_probability_adjustment(self) -> bool:
        pass

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[Any, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return {
            "system_geometry": self.system_geometry.get_log_info(),
            "initial_system_state": self.initial_system_state.get_log_info(),
            **additional_info,
        }


class MonteCarloSampler(GeneralSampler):
    def __init__(
        self,
        system_geometry: state.SystemGeometry,
        initial_system_state: state.InitialSystemState,
        system_hamiltonian: hamiltonian.Hamiltonian,
        num_samples: int,
        num_intermediate_mc_steps: int,
        state_modification: state.StateModification,
        num_thermalization_steps: int,
        num_samples_per_chain: int,
    ):
        super().__init__(
            system_geometry=system_geometry, initial_system_state=initial_system_state
        )
        self.system_hamiltonian = system_hamiltonian
        self.num_samples = num_samples
        self.num_intermediate_mc_steps = num_intermediate_mc_steps
        self.state_modification = state_modification
        self.num_thermalization_steps = num_thermalization_steps
        self.num_samples_per_chain = num_samples_per_chain

        print(
            f"Monte Carlo Sampling used. Approximately {self.num_samples} samples and {self.num_samples_per_chain} which means ca. {self.num_samples/self.num_samples_per_chain:.1f} chains will be run across all workers"
        )

    def accepts_modification(
        self,
        energy_difference: np.complex128,
        psi_factor: float,
        random_generator: RandomGenerator,
    ) -> bool:
        acceptance_ratio: float = float(
            np.real(psi_factor * np.exp(2 * np.real(energy_difference)))
        )

        # Accept or reject the proposed state
        return random_generator.probability() <= acceptance_ratio

    def do_metropolis_steps(
        self,
        state_to_modify: state.SystemState,
        num_steps: int,
        time: float,
        random_generator: RandomGenerator,
    ) -> None:
        """
        Advances the system_state in-place by num_steps metropolis steps
        """
        for _ in range(num_steps):
            if isinstance(self.state_modification, state.RandomFlipping):
                # Propose a new state by random modification
                original_state = state_to_modify
                proposed_state = self.state_modification.get_random_flipped_copy(
                    before_flip_system_state=state_to_modify,
                    random_generator=random_generator,
                )

                # Calculate the energies and probabilities
                original_state_psi = original_state.get_Psi_of_N()
                proposed_state_psi = proposed_state.get_Psi_of_N()

                psi_factor = float(
                    np.real(proposed_state_psi * np.conjugate(proposed_state_psi))
                    / np.real(original_state_psi * np.conjugate(original_state_psi))
                )

                # TODO use new specialized flipping function
                # also check for correct direction...

                energy_difference = self.system_hamiltonian.get_H_eff_difference(
                    system_state_a=proposed_state,  # make sure this does proposed-original !!
                    system_state_b=original_state,
                    time=time,
                )

                if self.accepts_modification(
                    energy_difference=energy_difference,
                    random_generator=random_generator,
                    psi_factor=psi_factor,
                ):
                    state_to_modify.set_state_array(
                        new_state=proposed_state.get_state_array()
                    )

            elif isinstance(self.state_modification, state.LatticeNeighborHopping):

                sw1_up, sw1_index, sw2_up, sw2_index = (
                    self.state_modification.get_lattice_hopping_parameters(
                        random_generator=random_generator,
                    )
                )

                energy_difference, psi_factor = (
                    # TODO is this the right orientation of the difference ???
                    self.system_hamiltonian.get_H_eff_difference_swapping(
                        time=time,
                        sw1_up=sw1_up,
                        sw1_index=sw1_index,
                        sw2_up=sw2_up,
                        sw2_index=sw2_index,
                        before_swap_system_state=state_to_modify,
                    )
                )

                if self.accepts_modification(
                    energy_difference=-energy_difference,  # difference is wrong way round from function. need proposed-original
                    random_generator=random_generator,
                    psi_factor=psi_factor,
                ):
                    state_to_modify.swap_in_place(
                        sw1_up=sw1_up,
                        sw1_index=sw1_index,
                        sw2_up=sw2_up,
                        sw2_index=sw2_index,
                    )
            else:
                raise Exception(
                    f"Handling case for state-modification {self.state_modification.__class__.__name__} not implemented"
                )

    def initialize_fill_level(
        self,
        state_to_modify: state.SystemState,
        random_generator: RandomGenerator,
    ):
        # For flipping the amount of initial filling is irrelevant, because it will change in thermalization
        # For swapping, this dictates the overall amount of particles present, therefore maps a specific block in the hamiltonian
        # The randomization is therefore relevant, to gauge which blocks are possibly sampled

        # The distribution of the fill level sampling (number of 1s) therefore needs to be binomial
        # This is equal to choosing 1/0 with equal probability for each site

        state_to_modify.init_random_filling(random_generator=random_generator)

        # TODO make the initialization fill level configurable from outside

    def thermalize(
        self,
        state_to_modify: state.SystemState,
        time: float,
        random_generator: RandomGenerator,
    ):
        self.do_metropolis_steps(
            state_to_modify,
            num_steps=self.num_thermalization_steps,
            time=time,
            random_generator=random_generator,
        )

    def sample_generator(
        self,
        time: float,
        worker_index: int,
        num_workers: int,
        random_generator: Union[RandomGenerator, None],
    ) -> Generator[state.SystemState, None, None]:
        if random_generator is None:
            raise Exception("Monte Carlo Sampler needs a random generator")
        else:
            _ = worker_index  # this generator is independent of worker_index

            number_of_chains = math.ceil(
                (self.num_samples / self.num_samples_per_chain) / num_workers
            )

            for chain_index in range(number_of_chains):
                if chain_index == number_of_chains - 1:
                    # truncate last chain to not overdo on samples
                    chain_target_count = math.ceil(
                        (self.num_samples / num_workers)
                        - ((number_of_chains - 1) * self.num_samples_per_chain)
                    )
                else:
                    chain_target_count = self.num_samples_per_chain

                working_state = state.SystemState(
                    system_geometry=self.system_geometry,
                    initial_system_state=self.initial_system_state,
                )

                self.initialize_fill_level(
                    working_state, random_generator=random_generator
                )
                self.thermalize(working_state, time, random_generator=random_generator)

                for _ in range(chain_target_count):
                    yield working_state
                    self.do_metropolis_steps(
                        working_state,
                        num_steps=self.num_intermediate_mc_steps,
                        time=time,
                        random_generator=random_generator,
                    )

    def produces_samples_count(self) -> int:
        return self.num_samples

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[Any, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[Any, Any]]]:

        return super().get_log_info(
            {
                "type": "MonteCarloSampler",
                "num_samples": self.num_samples,
                "num_intermediate_mc_steps": self.num_intermediate_mc_steps,
                "num_thermalization_steps": self.num_thermalization_steps,
                "num_samples_per_chain": self.num_samples_per_chain,
                "state_modification": self.state_modification.get_log_info(),
                **additional_info,
            }
        )

    def requires_probability_adjustment(self) -> bool:
        return False


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
        self,
        time: float,
        worker_index: int,
        num_workers: int,
        random_generator: Union[RandomGenerator, None],
    ) -> Generator[state.SystemState, None, None]:
        _ = time  # this generator is independent of time
        _ = random_generator  # this generator doesn't need a random generator

        working_state = state.SystemState(
            system_geometry=self.system_geometry,
            initial_system_state=self.initial_system_state,
        )
        array_length = working_state.get_state_array().shape[0]

        # Calculate the range of indices for this worker
        total_samples = self.all_samples_count()
        start_index = (total_samples // num_workers) * worker_index
        if worker_index == num_workers - 1:
            # Assign the remaining samples to the last worker
            end_index = total_samples
        else:
            end_index = (total_samples // num_workers) * (worker_index + 1)

        # initialize the working array
        initializer_array = working_state.get_state_array()
        for i in range(array_length):
            initializer_array[i] = (start_index >> i) & 1

        # produce the samples
        for _ in range(end_index - start_index):
            yield working_state

            carry = 1
            for i in range(array_length):
                res = working_state.get_state_array()[i] + carry

                working_state.get_state_array()[i] = res % 2
                carry = res // 2

    def produces_samples_count(self) -> int:
        return self.all_samples_count()

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[Any, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return super().get_log_info({"type": "ExactSampler", **additional_info})

    def requires_probability_adjustment(self) -> bool:
        return True
