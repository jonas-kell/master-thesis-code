import numpy as np
from abc import ABC, abstractmethod
from systemgeometry import SystemGeometry
from randomgenerator import RandomGenerator
from typing import Dict, Union, Any, List
import numpy.typing as npt


class InitialSystemState(ABC):
    def __init__(self, system_geometry: SystemGeometry):
        self.domain_size = system_geometry.get_number_sites_wo_spin_degree()

    @abstractmethod
    def get_Psi_of_N(self, system_state_array: npt.NDArray[np.uint8]) -> float:
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

    def get_Psi_of_N(self, system_state_array: npt.NDArray[np.uint8]) -> float:
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

    def get_Psi_of_N(self, system_state_array: npt.NDArray[np.uint8]) -> float:
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
        state_array: Union[npt.NDArray[np.uint8], None] = None,
    ):
        self.system_geometry = system_geometry
        self.initial_system_state = initial_system_state

        if state_array is None:
            self.state_array: npt.NDArray[np.uint8] = np.zeros(
                (self.system_geometry.get_number_sites(),), dtype=np.uint8
            )
        else:
            self.state_array = state_array

    def get_state_array(self) -> npt.NDArray[np.uint8]:
        return self.state_array

    def set_state_array(self, new_state: npt.NDArray[np.uint8]) -> None:
        self.state_array = new_state

    def get_random_flipped_copy(
        self, num_flips: int, random_generator: RandomGenerator
    ) -> "SystemState":
        all_sites = self.get_number_sites()
        new_state = self.state_array.copy()

        for _ in range(num_flips):
            flip_index = random_generator.randint(0, all_sites - 1)

            new_state[flip_index] = (self.state_array[flip_index] + 1) % 2

        return SystemState(
            system_geometry=self.system_geometry,
            initial_system_state=self.initial_system_state,
            state_array=new_state,
        )

    def get_editable_copy(self) -> "SystemState":
        new_state = self.state_array.copy()

        return SystemState(
            system_geometry=self.system_geometry,
            initial_system_state=self.initial_system_state,
            state_array=new_state,
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
            self.state_array = np.zeros_like(self.state_array)
            while added < target_num_filling:
                place_index = random_generator.randint(0, all_sites - 1)
                if self.state_array[place_index] == 0:
                    self.state_array[place_index] = 1
                    added += 1
        else:
            # init with ones and remove
            self.state_array = np.ones_like(self.state_array)
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
