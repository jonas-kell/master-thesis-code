from abc import ABC, abstractmethod
from typing import Dict, Union, Any
from state import SystemState
from systemgeometry import SystemGeometry, LinearChainNonPeriodicState
import numpy.typing as npt
import numpy as np


class PSISelection(ABC):
    def __init__(
        self,
        system_geometry: SystemGeometry,
        J: float,
    ):
        self.system_geometry = system_geometry
        self.J = J

    @abstractmethod
    def get_number_of_PSIs(self) -> int:
        pass

    @abstractmethod
    def eval_PSIs_on_state(self, system_state: SystemState) -> npt.NDArray[np.float64]:
        pass

    def eval_PSI_differences_double_flipping(
        self, before_swap_system_state: SystemState, l: int, m: int, spins_up: bool
    ) -> float:
        domain_size = self.system_geometry.get_number_sites_wo_spin_degree()

        if spins_up:
            use_index_l = l % domain_size
            use_index_m = m % domain_size
        else:
            use_index_l = self.system_geometry.get_opposite_spin_index(l % domain_size)
            use_index_m = self.system_geometry.get_opposite_spin_index(m % domain_size)

        system_state_copy = (
            before_swap_system_state.get_editable_copy()
        )  # TODO difference-optimization will not need copy
        state_array = system_state_copy.get_state_array()

        first_val = self.eval_PSIs_on_state(system_state=system_state_copy)
        state_array[[use_index_l, use_index_m]] = (
            1 - state_array[[use_index_l, use_index_m]]
        )  # double flip on copy
        second_val = self.eval_PSIs_on_state(system_state=system_state_copy)

        # Caution, in most cases, this needs would be needed inverted one more time
        return first_val - second_val

    # TODO: implement flip optimization for this (swapping optimization can be done with the function above)

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return {
            **additional_info,
            "number_of_psis": self.get_number_of_PSIs(),
            "J": self.J,
        }


class ChainDirectionDependentAllSameFirstOrder(PSISelection):
    def __init__(
        self,
        system_geometry: SystemGeometry,
        J: float,
    ):
        super().__init__(
            system_geometry=system_geometry,
            J=J,
        )

        if not isinstance(self.system_geometry, LinearChainNonPeriodicState):
            raise Exception("PSI Selection is only compatible with a chain-geometry")

    def get_number_of_PSIs(self) -> int:
        # one in each direction, for A,B,C parts of V
        return 6

    def eval_PSIs_on_state(self, system_state: SystemState) -> npt.NDArray[np.float64]:
        psi_evals = np.zeros(self.get_number_of_PSIs(), dtype=np.float64)

        domain_size = self.system_geometry.get_number_sites_wo_spin_degree()

        for l in range(domain_size):
            for m in self.system_geometry.get_nearest_neighbor_indices(l):
                # index add determines the right/left neighbor interaction
                if m > l:
                    index_add = 0
                else:
                    index_add = 3

                state_array = system_state.get_state_array()
                occ_l = state_array[l]
                occ_m = state_array[m]
                occ_l_os = state_array[l + domain_size]
                occ_m_os = state_array[m + domain_size]

                psi_evals[0 + index_add] += occ_l * (1 - occ_m) * (
                    occ_l_os == occ_m_os
                ) + occ_l_os * (1 - occ_m_os) * (occ_l == occ_m)
                psi_evals[1 + index_add] += occ_l * (1 - occ_m) * occ_l_os * (
                    1 - occ_m_os
                ) + occ_l_os * (1 - occ_m_os) * occ_l * (1 - occ_m)
                psi_evals[2 + index_add] += occ_l * (1 - occ_m) * occ_m_os * (
                    1 - occ_l_os
                ) + occ_l_os * (1 - occ_m_os) * occ_m * (1 - occ_l)

        psi_evals *= self.J  # all are just * J
        return psi_evals

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return super().get_log_info(
            {
                "type": "ChainDirectionDependentAllSameFirstOrder",
            }
        )
