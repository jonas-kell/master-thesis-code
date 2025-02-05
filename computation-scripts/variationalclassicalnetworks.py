from abc import ABC, abstractmethod
from typing import Dict, Union, Any, Tuple, List
from state import SystemState
from systemgeometry import (
    SystemGeometry,
    LinearChainNonPeriodicState,
    SquareSystemNonPeriodicState,
)
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

    def eval_PSIs_on_state(
        self,
        system_state: SystemState,
    ) -> npt.NDArray[np.complex128]:
        psi_evals = np.zeros(self.get_number_of_PSIs(), dtype=np.complex128)

        domain_size = self.system_geometry.get_number_sites_wo_spin_degree()

        state_array = system_state.get_state_array()
        for l in range(domain_size):

            occ_l = state_array[l]
            occ_l_os = state_array[l + domain_size]

            for m in self.system_geometry.get_nearest_neighbor_indices(l):
                occ_m = state_array[m]
                occ_m_os = state_array[m + domain_size]

                for contrib_index, contrib in self.PSI_contribution(
                    l=l,
                    m=m,
                    occ_l=occ_l,
                    occ_l_os=occ_l_os,
                    occ_m=occ_m,
                    occ_m_os=occ_m_os,
                ):
                    psi_evals[contrib_index] += contrib

        # all are just * J - this is not relevant for the learned parameters, but makes the learned parameter scale match the Lambda_A,B,C from that analytical calculation
        psi_evals *= self.J
        return psi_evals

    @abstractmethod
    def PSI_contribution(
        self,
        l: int,
        m: int,
        occ_l: int,
        occ_l_os: int,
        occ_m: int,
        occ_m_os: int,
    ) -> List[Tuple[int, np.complex128]]:
        pass

    def eval_PSI_differences_flipping_unoptimized(
        self, before_swap_system_state: SystemState, l: int, spin_up: bool
    ) -> npt.NDArray[np.complex128]:
        domain_size = self.system_geometry.get_number_sites_wo_spin_degree()

        if spin_up:
            use_index_l = l % domain_size
        else:
            use_index_l = self.system_geometry.get_opposite_spin_index(l % domain_size)

        system_state_copy = (
            before_swap_system_state.get_editable_copy()
        )  # difference-optimization will not need copy
        state_array = system_state_copy.get_state_array()

        first_val = self.eval_PSIs_on_state(system_state=system_state_copy)
        state_array[use_index_l] = 1 - state_array[use_index_l]  # flip on copy
        second_val = self.eval_PSIs_on_state(system_state=system_state_copy)

        # Caution, in most cases, this needs would be needed inverted one more time
        return first_val - second_val

    def eval_PSI_differences_double_flipping_unoptimized(
        self,
        before_swap_system_state: SystemState,
        l: int,
        m: int,
        spin_l_up: bool,
        spin_m_up: bool,
    ) -> npt.NDArray[np.complex128]:
        domain_size = self.system_geometry.get_number_sites_wo_spin_degree()

        if spin_l_up:
            use_index_l = l % domain_size
        else:
            use_index_l = self.system_geometry.get_opposite_spin_index(l % domain_size)

        if spin_m_up:
            use_index_m = m % domain_size
        else:
            use_index_m = self.system_geometry.get_opposite_spin_index(m % domain_size)

        system_state_copy = (
            before_swap_system_state.get_editable_copy()
        )  # difference-optimization will not need copy
        state_array = system_state_copy.get_state_array()

        first_val = self.eval_PSIs_on_state(system_state=system_state_copy)
        state_array[[use_index_l, use_index_m]] = (
            1 - state_array[[use_index_l, use_index_m]]
        )  # double flip on copy
        second_val = self.eval_PSIs_on_state(system_state=system_state_copy)

        # Caution, in most cases, this needs would be needed inverted one more time
        return first_val - second_val

    def eval_PSI_differences_flipping(
        self, before_swap_system_state: SystemState, l: int, spin_up: bool
    ) -> npt.NDArray[np.complex128]:
        psi_evals = np.zeros(self.get_number_of_PSIs(), dtype=np.complex128)

        domain_size = self.system_geometry.get_number_sites_wo_spin_degree()

        state_array = before_swap_system_state.get_state_array()

        occ_l = state_array[l % domain_size]
        occ_l_os = state_array[
            l % domain_size + domain_size
        ]  # doesn't use get_other_spin_index as it is terrrrribly slow

        for m in self.system_geometry.get_nearest_neighbor_indices(l):
            occ_m = state_array[m % domain_size]
            occ_m_os = state_array[
                m % domain_size + domain_size
            ]  # doesn't use get_other_spin_index as it is terrrrribly slow

            # before
            for res_index, val in self.PSI_contribution(
                l=l,
                m=m,
                occ_l=occ_l,
                occ_l_os=occ_l_os,
                occ_m=occ_m,
                occ_m_os=occ_m_os,
            ):
                psi_evals[res_index] += val
            for res_index, val in self.PSI_contribution(
                l=m,
                m=l,
                occ_l=occ_m,
                occ_l_os=occ_m_os,
                occ_m=occ_l,
                occ_m_os=occ_l_os,
            ):
                psi_evals[res_index] += val

            # after
            for res_index, val in self.PSI_contribution(
                l=l,
                m=m,
                occ_l=(1 - occ_l if spin_up else occ_l),
                occ_l_os=(1 - occ_l_os if not spin_up else occ_l_os),
                occ_m=occ_m,
                occ_m_os=occ_m_os,
            ):
                psi_evals[res_index] -= val
            for res_index, val in self.PSI_contribution(
                l=m,
                m=l,
                occ_l=occ_m,
                occ_l_os=occ_m_os,
                occ_m=(1 - occ_l if spin_up else occ_l),
                occ_m_os=(1 - occ_l_os if not spin_up else occ_l_os),
            ):
                psi_evals[res_index] -= val

        return psi_evals * self.J  # times J for consistency sake

    def eval_PSI_differences_double_flipping(
        self,
        before_swap_system_state: SystemState,
        l: int,
        m: int,
        spin_l_up: bool,
        spin_m_up: bool,
    ) -> npt.NDArray[np.complex128]:
        psi_evals = np.zeros(self.get_number_of_PSIs(), dtype=np.complex128)

        domain_size = self.system_geometry.get_number_sites_wo_spin_degree()

        state_array = before_swap_system_state.get_state_array()

        occ_l = state_array[l % domain_size]
        occ_l_os = state_array[
            l % domain_size + domain_size
        ]  # doesn't use get_other_spin_index as it is terrrrribly slow
        occ_m = state_array[m % domain_size]
        occ_m_os = state_array[
            m % domain_size + domain_size
        ]  # doesn't use get_other_spin_index as it is terrrrribly slow

        for sw_index, sw_up, other_index, other_up, occ_sw, occ_sw_os in [
            (l, spin_l_up, m, spin_m_up, occ_l, occ_l_os),
            (m, spin_m_up, l, spin_l_up, occ_m, occ_m_os),
        ]:
            for nb_index in self.system_geometry.get_nearest_neighbor_indices(sw_index):
                occ_nb = state_array[nb_index % domain_size]
                occ_nb_os = state_array[
                    nb_index % domain_size + domain_size
                ]  # doesn't use get_other_spin_index as it is terrrrribly slow

                # before
                for res_index, val in self.PSI_contribution(
                    l=sw_index,
                    m=nb_index,
                    occ_l=occ_sw,
                    occ_l_os=occ_sw_os,
                    occ_m=occ_nb,
                    occ_m_os=occ_nb_os,
                ):
                    psi_evals[res_index] += val
                if other_index != nb_index:
                    # avoid double counting
                    for res_index, val in self.PSI_contribution(
                        l=nb_index,
                        m=sw_index,
                        occ_l=occ_nb,
                        occ_l_os=occ_nb_os,
                        occ_m=occ_sw,
                        occ_m_os=occ_sw_os,
                    ):
                        psi_evals[res_index] += val

                # after
                for res_index, val in self.PSI_contribution(
                    l=sw_index,
                    m=nb_index,
                    occ_l=(1 - occ_sw if sw_up else occ_sw),
                    occ_l_os=(1 - occ_sw_os if not sw_up else occ_sw_os),
                    occ_m=(
                        1 - occ_nb if nb_index == other_index and other_up else occ_nb
                    ),
                    occ_m_os=(
                        1 - occ_nb_os
                        if nb_index == other_index and not other_up
                        else occ_nb_os
                    ),
                ):
                    psi_evals[res_index] -= val
                if other_index != nb_index:
                    # avoid double counting
                    for res_index, val in self.PSI_contribution(
                        l=nb_index,
                        m=sw_index,
                        occ_l=occ_nb,  # here it is already not possible to have overlap
                        occ_l_os=occ_nb_os,
                        occ_m=(1 - occ_sw if sw_up else occ_sw),
                        occ_m_os=(1 - occ_sw_os if not sw_up else occ_sw_os),
                    ):
                        psi_evals[res_index] -= val

        return psi_evals * self.J  # times J for consistency sake

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

    def PSI_contribution(
        self,
        l: int,
        m: int,
        occ_l: int,
        occ_l_os: int,
        occ_m: int,
        occ_m_os: int,
    ) -> List[Tuple[int, np.complex128]]:

        # index add determines the right/left neighbor interaction
        if m > l:
            index_add = 0
        else:
            index_add = 3

        return [
            (
                0 + index_add,
                occ_l * (1 - occ_m) * (occ_l_os == occ_m_os)
                + occ_l_os * (1 - occ_m_os) * (occ_l == occ_m),
            ),
            (
                1 + index_add,
                occ_l * (1 - occ_m) * occ_l_os * (1 - occ_m_os)
                + occ_l_os * (1 - occ_m_os) * occ_l * (1 - occ_m),
            ),
            (
                2 + index_add,
                occ_l * (1 - occ_m) * occ_m_os * (1 - occ_l_os)
                + occ_l_os * (1 - occ_m_os) * occ_m * (1 - occ_l),
            ),
        ]

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return super().get_log_info(
            {
                "type": "ChainDirectionDependentAllSameFirstOrder",
            }
        )


class ChainNotDirectionDependentAllSameFirstOrder(PSISelection):
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
        # one for A,B,C parts of V
        return 3

    def PSI_contribution(
        self,
        l: int,
        m: int,
        occ_l: int,
        occ_l_os: int,
        occ_m: int,
        occ_m_os: int,
    ) -> List[Tuple[int, np.complex128]]:

        return [
            (
                0,
                (occ_l * (1 - occ_m) * (occ_l_os == occ_m_os))
                + (occ_l_os * (1 - occ_m_os) * (occ_l == occ_m)),
            ),
            (
                1,
                (occ_l * (1 - occ_m) * occ_l_os * (1 - occ_m_os))
                + (occ_l_os * (1 - occ_m_os) * occ_l * (1 - occ_m)),
            ),
            (
                2,
                occ_l * (1 - occ_m) * occ_m_os * (1 - occ_l_os)
                + (occ_l_os * (1 - occ_m_os) * occ_m * (1 - occ_l)),
            ),
        ]

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return super().get_log_info(
            {
                "type": "ChainNotDirectionDependentAllSameFirstOrder",
            }
        )


class SquareDirectionDependentAllSameFirstOrder(PSISelection):
    def __init__(
        self,
        system_geometry: SystemGeometry,
        J: float,
    ):
        super().__init__(
            system_geometry=system_geometry,
            J=J,
        )

        if not isinstance(self.system_geometry, SquareSystemNonPeriodicState):
            raise Exception("PSI Selection is only compatible with a square-geometry")
        else:
            # do not need to take the root, as we have identified the calss we can direct access it
            self.square_side_length = self.system_geometry.size

    def get_number_of_PSIs(self) -> int:
        # one in each direction, for A,B,C parts of V
        return 12

    def PSI_contribution(
        self,
        l: int,
        m: int,
        occ_l: int,
        occ_l_os: int,
        occ_m: int,
        occ_m_os: int,
    ) -> List[Tuple[int, np.complex128]]:

        is_vertical = (l % self.square_side_length) == (m % self.square_side_length)

        if not is_vertical:
            # index add determines the right/left neighbor interaction
            if m > l:
                index_add = 0
            else:
                index_add = 3
        else:
            # index add determines the up/down neighbor interaction
            if m > l:
                index_add = 6
            else:
                index_add = 9

        return [
            (
                0 + index_add,
                occ_l * (1 - occ_m) * (occ_l_os == occ_m_os)
                + occ_l_os * (1 - occ_m_os) * (occ_l == occ_m),
            ),
            (
                1 + index_add,
                occ_l * (1 - occ_m) * occ_l_os * (1 - occ_m_os)
                + occ_l_os * (1 - occ_m_os) * occ_l * (1 - occ_m),
            ),
            (
                2 + index_add,
                occ_l * (1 - occ_m) * occ_m_os * (1 - occ_l_os)
                + occ_l_os * (1 - occ_m_os) * occ_m * (1 - occ_l),
            ),
        ]

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return super().get_log_info(
            {
                "type": "SquareDirectionDependentAllSameFirstOrder",
            }
        )
