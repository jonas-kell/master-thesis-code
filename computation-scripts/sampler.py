from typing import Dict, Union, Any, Generator
from abc import ABC, abstractmethod
import math
import hamiltonian
from randomgenerator import RandomGenerator
import state
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


class BeforeThermalizationRandomization(ABC):
    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def prepare_state_for_thermalization(
        self,
        system_state: state.SystemState,
        random_generator: RandomGenerator,
    ) -> state.SystemState:
        pass

    @abstractmethod
    def get_log_info(self) -> Dict[str, Union[float, str]]:
        pass


class VacuumStateBeforeThermalization(BeforeThermalizationRandomization):
    def __init__(
        self,
    ):
        pass

    def prepare_state_for_thermalization(
        self,
        system_state: state.SystemState,
        random_generator: RandomGenerator,
    ) -> state.SystemState:
        system_state.get_state_array().fill(0)
        return system_state

    def get_log_info(self) -> Dict[str, Union[float, str]]:
        return {"type": "VacuumStateBeforeThermalization"}


class EachSiteRandomBeforeThermalization(BeforeThermalizationRandomization):
    def __init__(
        self,
    ):
        pass

    def prepare_state_for_thermalization(
        self,
        system_state: state.SystemState,
        random_generator: RandomGenerator,
    ) -> state.SystemState:
        system_state.init_random_filling(random_generator=random_generator)
        return system_state

    def get_log_info(self) -> Dict[str, Union[float, str]]:
        return {"type": "EachSiteRandomBeforeThermalization"}


class FillRandomlyToSpecifiedFillLevelBeforeThermalization(
    BeforeThermalizationRandomization
):
    def __init__(
        self,
        fill_ratio: float,
    ):
        if fill_ratio < 0:
            raise Exception("Fill ratio must be at least 0")

        if fill_ratio > 1:
            raise Exception("Fill ratio must be at most 1")

        self.fill_ratio = fill_ratio

    def prepare_state_for_thermalization(
        self,
        system_state: state.SystemState,
        random_generator: RandomGenerator,
    ) -> state.SystemState:
        system_state.fill_randomly_to_fill_level(
            fill_ratio=self.fill_ratio, random_generator=random_generator
        )
        return system_state

    def get_log_info(self) -> Dict[str, Union[float, str]]:
        return {
            "type": "FillRandomlyToSpecifiedFillLevelBeforeThermalization",
            "fill_ratio": self.fill_ratio,
        }


class FillRandomlyToFillLevelPulledFromUniformDistributionBeforeThermalization(
    BeforeThermalizationRandomization
):
    def __init__(
        self,
    ):
        pass

    def prepare_state_for_thermalization(
        self,
        system_state: state.SystemState,
        random_generator: RandomGenerator,
    ) -> state.SystemState:
        system_state.fill_randomly_to_fill_level(
            fill_ratio=random_generator.probability(), random_generator=random_generator
        )
        return system_state

    def get_log_info(self) -> Dict[str, Union[float, str]]:
        return {
            "type": "FillRandomlyToFillLevelPulledFromUniformDistributionBeforeThermalization",
        }


class FillRandomlyToFillLevelPulledFromBinomialDistributionBeforeThermalization(
    BeforeThermalizationRandomization
):
    """
    This should have the same outcome as EachSiteRandomBeforeThermalization

    Only that this one is less efficient in generating the state.
    So probably use the other one in long runs.
    """

    def __init__(
        self,
    ):
        pass

    def prepare_state_for_thermalization(
        self,
        system_state: state.SystemState,
        random_generator: RandomGenerator,
    ) -> state.SystemState:
        number_sites = system_state.get_number_sites()
        system_state.fill_randomly_to_fill_level(
            fill_ratio=random_generator.binomial_random(number_sites) / number_sites,
            random_generator=random_generator,
        )
        return system_state

    def get_log_info(self) -> Dict[str, Union[float, str]]:
        return {
            "type": "FillRandomlyToFillLevelPulledFromBinomialDistributionBeforeThermalization",
        }


class MonteCarloSampler(GeneralSampler):
    def __init__(
        self,
        system_geometry: state.SystemGeometry,
        initial_system_state: state.InitialSystemState,
        num_samples: int,
        num_intermediate_mc_steps: int,
        state_modification: state.StateModification,
        state_modification_thermalization: Union[state.StateModification, None],
        num_thermalization_steps: int,
        num_samples_per_chain: int,
        before_thermalization_initialization: BeforeThermalizationRandomization,
    ):
        super().__init__(
            system_geometry=system_geometry, initial_system_state=initial_system_state
        )
        self.system_hamiltonian = None
        self.num_samples = num_samples
        self.num_intermediate_mc_steps = num_intermediate_mc_steps
        self.state_modification = state_modification
        self.num_thermalization_steps = num_thermalization_steps
        self.num_samples_per_chain = num_samples_per_chain
        self.before_thermalization_initialization = before_thermalization_initialization

        if state_modification_thermalization is None:
            self.state_modification_thermalization = self.state_modification
        else:
            self.state_modification_thermalization = state_modification_thermalization

        print(
            f"Monte Carlo Sampling used. Approximately {self.num_samples} samples and {self.num_samples_per_chain} which means ca. {self.num_samples/self.num_samples_per_chain:.1f} chains will be run across all workers"
        )

    def init_hamiltonian(
        self,
        system_hamiltonian: hamiltonian.Hamiltonian,
    ):
        """In initialization it is otherwise notpossible to init hamiltonian and sampler, because they cross-reference"""
        self.system_hamiltonian = system_hamiltonian

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
        state_modification: state.StateModification,
        state_to_modify: state.SystemState,
        num_steps: int,
        time: float,
        random_generator: RandomGenerator,
    ) -> None:
        """
        Advances the system_state in-place by num_steps metropolis steps
        """

        if self.system_hamiltonian is None:
            raise Exception("Need to call 'init_hamiltonian' before using!!")

        for _ in range(num_steps):
            if isinstance(state_modification, state.RandomFlipping):
                # Propose a new state modification
                (
                    flipping_up,
                    flipping_index,
                ) = state_modification.get_random_flipping_parameters(  # these are in range 0<=index<#_wo_spin_degree
                    random_generator=random_generator
                )

                energy_difference, psi_factor = (
                    self.system_hamiltonian.get_H_eff_difference_flipping(
                        time=time,
                        flipping_up=flipping_up,
                        flipping_index=flipping_index,
                        before_swap_system_state=state_to_modify,
                    )
                )

                if self.accepts_modification(
                    energy_difference=-energy_difference,  # notice minus: difference is wrong way round from function. need (proposed - original)
                    random_generator=random_generator,
                    psi_factor=1
                    / psi_factor,  # same reason as above also inverts this factor
                ):
                    state_to_modify.flip_in_place(
                        flipping_up=flipping_up, flipping_index=flipping_index
                    )

            elif isinstance(state_modification, state.LatticeNeighborHopping):

                (
                    sw1_up,
                    sw1_index,
                    sw2_up,
                    sw2_index,
                ) = state_modification.get_lattice_hopping_parameters(  # these are in range 0<=index<#_wo_spin_degree
                    random_generator=random_generator,
                )

                energy_difference, psi_factor = (
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
                    energy_difference=-energy_difference,  # notice minus: difference is wrong way round from function. need (proposed - original)
                    random_generator=random_generator,
                    psi_factor=1
                    / psi_factor,  # same reason as above also inverts this factor
                ):
                    state_to_modify.swap_in_place(
                        sw1_up=sw1_up,
                        sw1_index=sw1_index,
                        sw2_up=sw2_up,
                        sw2_index=sw2_index,
                    )
            else:
                raise Exception(
                    f"Handling case for state-modification {state_modification.__class__.__name__} not implemented"
                )

    def prepare_for_thermalization(
        self,
        state_to_modify: state.SystemState,
        random_generator: RandomGenerator,
    ):
        # For flipping the amount of initial filling is irrelevant, because it will change in thermalization
        # For swapping, this dictates the overall amount of particles present, therefore maps a specific block in the hamiltonian
        # The randomization is therefore relevant, to gauge which blocks are possibly sampled

        # Assumption:
        # The distribution of the fill level sampling (number of 1s) therefore needs to be binomial
        # This is equal to choosing 1/0 with equal probability for each site

        self.before_thermalization_initialization.prepare_state_for_thermalization(
            system_state=state_to_modify,
            random_generator=random_generator,
        )

    def thermalize(
        self,
        state_to_modify: state.SystemState,
        time: float,
        random_generator: RandomGenerator,
    ):
        self.do_metropolis_steps(
            state_to_modify=state_to_modify,
            state_modification=self.state_modification_thermalization,
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

                self.prepare_for_thermalization(
                    working_state, random_generator=random_generator
                )
                self.thermalize(working_state, time, random_generator=random_generator)

                for _ in range(chain_target_count):
                    yield working_state
                    self.do_metropolis_steps(
                        state_to_modify=working_state,
                        state_modification=self.state_modification_thermalization,
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
                "state_modification_thermalization": self.state_modification_thermalization.get_log_info(),
                "pre_thermalization_initialization_strategy": self.before_thermalization_initialization.get_log_info(),
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

        # make sure, to start the sampling
        # from 1,1,1,1,1
        # then 1,1,1,1,0
        # then 1,1,1,0,1
        # ....
        # last 0,0,0,0,0

        # because that way, the sampling order is equivalent to the natural ordering
        # up,up,up,up,up
        # up,up,up,up,down <- these up/down have NOTHING to do with the up/down-degree of the hc-bosons, but stem from the mapping to spins to generate the density matrix equivalent
        # ...
        # down,... this is required, to match the logical sigma_y convention

        # initialize the working array
        initializer_array = working_state.get_state_array()
        for i in range(array_length):
            # 1- to start 1111, not 0000, then shift the index and extract the required index
            initializer_array[array_length - i - 1] = 1 - ((start_index >> i) & 1)

        # produce the samples
        for _ in range(end_index - start_index):
            yield working_state

            carry = 1
            for i in range(array_length):
                working_index = array_length - i - 1
                res = (1 - working_state.get_state_array()[working_index]) + carry
                working_state.get_state_array()[working_index] = 1 - (res % 2)
                carry = res // 2

    def produces_samples_count(self) -> int:
        return self.all_samples_count()

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[Any, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return super().get_log_info({"type": "ExactSampler", **additional_info})

    def requires_probability_adjustment(self) -> bool:
        return True
