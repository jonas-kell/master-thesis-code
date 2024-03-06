import numpy as np
from abc import ABC, abstractmethod
from systemgeometry import SystemGeometry
from randomgenerator import RandomGenerator
from typing import List
from typing import Union


class InitialSystemState(ABC):
    def __init__(self, system_geometry: SystemGeometry):
        self.domain_size = system_geometry.get_number_sites_wo_spin_degree()

    @abstractmethod
    def get_Psi_of_N(self, system_state_array: np.ndarray) -> float:
        pass


class HomogenousInitialSystemState(InitialSystemState):
    def __init__(self, system_geometry: SystemGeometry):
        super().__init__(system_geometry)

        self.cached_answer: float = 1 / (
            2 ** system_geometry.get_number_sites_wo_spin_degree()
        )

    def get_Psi_of_N(self, system_state_array: np.ndarray) -> float:
        system_state_array  # get rid of unused error, sorry

        return self.cached_answer


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

    def get_Psi_of_N(self, system_state_array: np.ndarray) -> float:
        # pre-mature optimization to save on case, but this is a
        # answer=0 -> pre_factor_not_important_case
        # answer=1 -> pre_factor_important_case

        # probably switch would be faster, with branch prediction, as we take the same case in nearly ALL situations and these operations waste many cycles because python

        return (
            self.pre_factor_not_important_case
            + (
                np.count_nonzero(system_state_array) == 2
                and system_state_array[self.site] == 1
                and system_state_array[self.site_os] == 1
            )
            * self.additional_for_important_case
        )


class SystemState:
    def __init__(
        self,
        system_geometry: SystemGeometry,
        initial_system_state: InitialSystemState,
        state: Union[np.ndarray, None] = None,
    ):
        self.system_geometry = system_geometry
        self.initial_system_state = initial_system_state

        if state is None:
            self.state = np.zeros(
                (self.system_geometry.get_number_sites(),), dtype=np.uint8
            )
        else:
            self.state = state

    def get_state_array(self) -> np.ndarray:
        return self.state

    def set_state_array(self, new_state: np.ndarray) -> None:
        self.state = new_state

    def get_random_flipped_copy(
        self, num_flips: int, random_generator: RandomGenerator
    ) -> "SystemState":
        all_sites = self.get_number_sites()
        new_state = self.state.copy()

        for _ in range(num_flips):
            flip_index = random_generator.randint(0, all_sites - 1)

            new_state[flip_index] = (self.state[flip_index] + 1) % 2

        return SystemState(
            system_geometry=self.system_geometry,
            initial_system_state=self.initial_system_state,
            state=new_state,
        )

    def get_editable_copy(self) -> "SystemState":
        new_state = self.state.copy()

        return SystemState(
            system_geometry=self.system_geometry,
            initial_system_state=self.initial_system_state,
            state=new_state,
        )

    def init_random_filling(
        self, fill_ratio: float, random_generator: RandomGenerator
    ) -> None:
        if fill_ratio < 0:
            raise Exception("Fill ratio must be at least 0")

        if fill_ratio > 1:
            raise Exception("Fill ratio must be at most 1")

        all_sites = self.get_number_sites()
        target_num_filling = int(all_sites * fill_ratio)

        if fill_ratio <= 0.5:
            # init with zeros and add
            added = 0
            self.state = np.zeros_like(self.state)
            while added < target_num_filling:
                place_index = random_generator.randint(0, all_sites - 1)
                if self.state[place_index] == 0:
                    self.state[place_index] = 1
                    added += 1
        else:
            # init with ones and remove
            self.state = np.ones_like(self.state)
            removed = 0
            target_num_removing = all_sites - target_num_filling
            while removed < target_num_removing:
                place_index = random_generator.randint(0, all_sites - 1)
                if self.state[place_index] == 1:
                    self.state[place_index] = 0
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
