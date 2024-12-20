from abc import ABC, abstractmethod
from typing import Dict, Union, Any, List, Tuple
from systemgeometry import SystemGeometry
from randomgenerator import RandomGenerator
import numpy.typing as npt
import numpy as np


class InitialSystemState(ABC):
    def __init__(self, system_geometry: SystemGeometry):
        self.domain_size = system_geometry.get_number_sites_wo_spin_degree()

    @abstractmethod
    def get_Psi_of_N(self, system_state_array: npt.NDArray[np.int8]) -> float:
        pass

    @abstractmethod
    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        pass


class HomogenousInitialSystemState(InitialSystemState):
    def __init__(self, system_geometry: SystemGeometry):
        super().__init__(system_geometry)

        self.cached_answer: float = 1 / (
            2 ** system_geometry.get_number_sites_wo_spin_degree()
        )

    def get_Psi_of_N(self, system_state_array: npt.NDArray[np.int8]) -> float:
        _ = system_state_array  # get rid of unused error

        return self.cached_answer

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return {"type": "HomogenousInitialSystemState"}


class SingularDoubleOccupationInitialSystemState(InitialSystemState):
    def __init__(
        self,
        site: int,
        more_important_factor: float,
        system_geometry: SystemGeometry,
    ):
        super().__init__(system_geometry)

        if site < 0:
            raise Exception("Site must be at least 0")

        if site >= self.domain_size:
            raise Exception(f"Site must be smaller than {self.domain_size} to fit")

        self.site = site
        self.site_os = system_geometry.get_opposite_spin_index(site)

        number_states = 2 ** system_geometry.get_number_sites()
        pre_factor_important_case = np.sqrt(
            1.0 / (1.0 + (number_states - 1.0) * (1.0 / more_important_factor**2))
        )
        pre_factor_not_important_case = (
            pre_factor_important_case / more_important_factor
        )

        self.pre_factor_not_important_case = pre_factor_not_important_case
        self.additional_for_important_case = (
            pre_factor_important_case - pre_factor_not_important_case
        )

    def get_Psi_of_N(self, system_state_array: npt.NDArray[np.int8]) -> float:
        # pre-mature optimization to save on case, but this is a
        # answer=0 -> pre_factor_not_important_case
        # answer=1 -> pre_factor_important_case

        # probably switch would be faster, with branch prediction, as we take the same case in nearly ALL situations and these operations waste many cycles because python

        return (
            self.pre_factor_not_important_case
            + (
                np.count_nonzero(system_state_array) == 2  # type: ignore -> function not properly typed. but input is ndarray, which works in this configuration
                and system_state_array[self.site] == 1
                and system_state_array[self.site_os] == 1
            )
            * self.additional_for_important_case
        )

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return {"type": "SingularDoubleOccupationInitialSystemState", "site": self.site}


class SystemState:
    def __init__(
        self,
        system_geometry: SystemGeometry,
        initial_system_state: InitialSystemState,
        state_array: Union[npt.NDArray[np.int8], None] = None,
    ):
        self.system_geometry = system_geometry
        self.initial_system_state = initial_system_state

        if state_array is None:
            self.state_array: npt.NDArray[np.int8] = np.zeros(
                (self.system_geometry.get_number_sites(),), dtype=np.int8
            )
        else:
            self.state_array = state_array

    def get_state_array(self) -> npt.NDArray[np.int8]:
        return self.state_array

    def set_state_array(self, new_state: npt.NDArray[np.int8]) -> None:
        self.state_array = new_state

    def get_editable_copy(self) -> "SystemState":
        new_state = self.state_array.copy()

        return SystemState(
            system_geometry=self.system_geometry,
            initial_system_state=self.initial_system_state,
            state_array=new_state,
        )

    def init_random_filling(self, random_generator: RandomGenerator) -> None:
        """
        Each site occupation is chosen randomly with a 50%/50% chance for occupied/not occupied.

        This is equivalent to placing a random number (from 0 to get_number_sites()) of 1s on random locations of the vacuum-state,
        where said random number (=fill-level)is chosen from a binomial distribution.
        """
        self.set_state_array(
            np.array(
                random_generator.rand_occupation_array(self.get_number_sites()),
                dtype=np.int8,
            )
        )

    def fill_randomly_to_fill_level(
        self, fill_ratio: float, random_generator: RandomGenerator
    ) -> None:

        all_sites = self.get_number_sites()
        target_num_filling = round(all_sites * fill_ratio)

        # TODO this is not fully efficient, also causes a random number of calls to the rng, which is kind of not pretty

        if fill_ratio <= 0.5:
            # init with zeros and add
            added = 0
            self.state_array.fill(0)
            while added < target_num_filling:
                place_index = random_generator.randint(0, all_sites - 1)
                if self.state_array[place_index] == 0:
                    self.state_array[place_index] = 1
                    added += 1
        else:
            # init with ones and remove
            self.state_array.fill(1)
            removed = 0
            target_num_removing = all_sites - target_num_filling
            while removed < target_num_removing:
                place_index = random_generator.randint(0, all_sites - 1)
                if self.state_array[place_index] == 1:
                    self.state_array[place_index] = 0
                    removed += 1

    # !! Function access helpers to internal system_geometry and initial_system_state from here on

    def get_Psi_of_N(self) -> float:
        return self.initial_system_state.get_Psi_of_N(self.get_state_array())

    def get_eps_multiplier(
        self, index: int, phi: float, sin_phi: float, cos_phi: float
    ) -> float:
        return self.system_geometry.get_eps_multiplier(
            index=index,
            phi=phi,
            sin_phi=sin_phi,
            cos_phi=cos_phi,
        )

    def get_nearest_neighbor_indices(self, index: int) -> List[int]:
        return self.system_geometry.get_nearest_neighbor_indices(index=index)

    def get_opposite_spin_index(self, index: int) -> int:
        return self.system_geometry.get_opposite_spin_index(index=index)

    def get_number_sites(self) -> int:
        return self.system_geometry.get_number_sites()

    def get_number_sites_wo_spin_degree(self) -> int:
        return self.system_geometry.get_number_sites_wo_spin_degree()

    def swap_in_place(
        self,
        sw1_up: bool,
        sw1_index: int,
        sw2_up: bool,
        sw2_index: int,
    ) -> "SystemState":
        # check that no unexpected too large indices are requested (in-place modification is not used much, so not too bad)
        domain_size = self.get_number_sites_wo_spin_degree()
        if (
            sw1_index < 0
            or sw2_index < 0
            or sw1_index >= domain_size
            or sw2_index >= domain_size
        ):
            raise Exception(
                f"Site must be bigger than 0 and smaller than {domain_size} to fit: {sw1_index} {sw2_index}"
            )

        use_swap_index_1 = sw1_index
        if not sw1_up:
            use_swap_index_1 = self.get_opposite_spin_index(sw1_index)
        use_swap_index_2 = sw2_index
        if not sw2_up:
            use_swap_index_2 = self.get_opposite_spin_index(sw2_index)

        self.get_state_array()[
            [use_swap_index_1, use_swap_index_2]
        ] = self.get_state_array()[[use_swap_index_2, use_swap_index_1]]

        return self

    def flip_in_place(
        self,
        flipping_up: bool,
        flipping_index: int,
    ) -> "SystemState":
        # check that no unexpected too large indices are requested (in-place modification is not used much, so not too bad)
        domain_size = self.get_number_sites_wo_spin_degree()
        if flipping_index < 0 or flipping_index >= domain_size:
            raise Exception(
                f"Site must be bigger than 0 and smaller than {domain_size} to fit: {flipping_index}"
            )

        use_flip_index = flipping_index
        if not flipping_up:
            use_flip_index = self.get_opposite_spin_index(flipping_index)

        self.get_state_array()[use_flip_index] = (
            1 - self.get_state_array()[use_flip_index]
        )

        return self


class StateModification(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        pass


class RandomFlipping(StateModification):
    def __init__(
        self,
        system_geometry: SystemGeometry,
    ):
        super().__init__()
        self.randomizer_upper_argument = (
            system_geometry.get_number_sites_wo_spin_degree() - 1
        )

    def get_random_flipping_parameters(
        self, random_generator: RandomGenerator
    ) -> Tuple[
        bool,  # flipping_up
        int,  # flipping_index
    ]:
        return (
            random_generator.randbool(),
            random_generator.randint(0, self.randomizer_upper_argument),
        )

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return {"type": "RandomFlipping"}


class LatticeNeighborHopping(StateModification):
    def __init__(
        self,
        allow_hopping_across_spin_direction: bool,
        system_geometry: SystemGeometry,
    ):
        super().__init__()
        self.allow_hopping_across_spin_direction = allow_hopping_across_spin_direction

        num_sites_wo_spin_degree = system_geometry.get_number_sites_wo_spin_degree()
        self.swap_cache: List[Tuple[int, int]] = []

        for i in range(num_sites_wo_spin_degree):
            for j in system_geometry.get_nearest_neighbor_indices(i):
                self.swap_cache.append((i, j))
                self.swap_cache.append((j, i))

        self.upper_bound_to_give_in_randomizer = len(self.swap_cache) - 1

    # DO NOT CHECK, THAT THIS CHANGES SOMETHING (This would force state changes and mess with the fluctuations/average)
    def get_lattice_hopping_parameters(
        self, random_generator: RandomGenerator
    ) -> Tuple[
        bool,  # sw1_up
        int,  # sw1_index
        bool,  # sw2_up
        int,  # sw2_index
    ]:
        sw1_up = random_generator.randbool()

        if self.allow_hopping_across_spin_direction:
            sw2_up = random_generator.randbool()
        else:
            sw2_up = sw1_up

        # !! CAUTION: It is not trivial, to generate a truly random swap
        # to make every SWAP evenly likely, you need to select one swappable edge from the list of swappable edges evenly likely.
        # This is NOT the same, as selecting an index randomly and then one of the adjacent indices
        # because if the first choice falls on a corner/edge site, it is more likely to hop inward, because edge sites have less neighbors

        sw1_index, sw2_index = self.swap_cache[
            random_generator.randint(0, self.upper_bound_to_give_in_randomizer)
        ]  # index is NOT increased depending on spin direction. Analytical calculation takes care of that/requires this this way

        return (
            sw1_up,
            sw1_index,
            sw2_up,
            sw2_index,
        )

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return {
            "type": "LatticeNeighborHopping",
            "allow_hopping_across_spin_direction": self.allow_hopping_across_spin_direction,
        }
