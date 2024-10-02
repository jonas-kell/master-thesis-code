from typing import Dict, List, Tuple, Callable, Union, Any
from enum import Enum
from abc import ABC, abstractmethod
import state
import numpy as np
from vcomponents import (
    v as calculate_v_plain,
    v_flip as calculate_v_flip,
    v_hop as calculate_v_hop,
)


class Hamiltonian(ABC):
    def __init__(
        self,
        U: float,
        E: float,
        J: float,
        phi: float,
    ):
        self.U = U
        self.E = E
        self.J = J
        self.phi = phi
        self.cos_phi: float = np.cos(self.phi)
        self.sin_phi: float = np.sin(self.phi)

    @abstractmethod
    def get_base_energy(
        self,
        system_state: state.SystemState,
    ) -> float:
        pass

    @abstractmethod
    def get_H_n(
        self,
        time: float,
        system_state: state.SystemState,
    ) -> np.complex128:
        pass

    def get_H_eff(
        self,
        time: float,
        system_state: state.SystemState,
    ) -> np.complex128:
        H_n = self.get_H_n(time=time, system_state=system_state)
        E_zero_n = self.get_base_energy(system_state=system_state)

        return H_n - (1j * E_zero_n * time)

    def get_H_eff_difference(
        self,
        time: float,
        system_state_a: state.SystemState,
        system_state_b: state.SystemState,
    ) -> np.complex128:
        return self.get_H_eff(time=time, system_state=system_state_a) - self.get_H_eff(
            time=time, system_state=system_state_b
        )

    def get_H_eff_difference_swapping(
        self,
        time: float,
        sw1_up: bool,
        sw1_index: int,
        sw2_up: bool,
        sw2_index: int,
        before_swap_system_state: state.SystemState,  # required, because un-optimized implementation uses state and optimized implementation uses it to pre-compute the occupations and lambda functions
    ) -> Tuple[np.complex128, float]:
        sw1_occupation = before_swap_system_state.get_state_array()[sw1_index]
        sw1_occupation_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(sw1_index)
        ]
        sw2_occupation = before_swap_system_state.get_state_array()[sw2_index]
        sw2_occupation_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(sw2_index)
        ]

        if sw1_occupation * sw1_up + sw1_occupation_os * (
            not sw1_up
        ) == sw2_occupation * sw2_up + sw2_occupation_os * (not sw2_up):
            # The swapped indices are equal. We know the result
            return (np.complex128(0), 1.0)

        # allocate swapped state
        after_swap_system_state = before_swap_system_state.get_editable_copy()
        after_swap_system_state.swap_in_place(
            sw1_up=sw1_up,
            sw1_index=sw1_index,
            sw2_up=sw2_up,
            sw2_index=sw2_index,
        )

        original_state_psi = before_swap_system_state.get_Psi_of_N()
        proposed_state_psi = after_swap_system_state.get_Psi_of_N()
        psi_factor = float(
            np.real(proposed_state_psi * np.conjugate(proposed_state_psi))
            / np.real(original_state_psi * np.conjugate(original_state_psi))
        )

        # return un-optimized default
        return (
            self.get_H_eff_difference(
                time=time,
                system_state_a=before_swap_system_state,
                system_state_b=after_swap_system_state,
            ),
            psi_factor,
        )

    def get_H_eff_difference_flipping(
        self,
        time: float,
        flipping_up: bool,
        flipping_index: int,
        before_swap_system_state: state.SystemState,  # required, because un-optimized implementation uses state and optimized implementation uses it to pre-compute the occupations and lambda functions
    ) -> Tuple[np.complex128, float]:
        # allocate swapped state
        after_swap_system_state = before_swap_system_state.get_editable_copy()
        after_swap_system_state.flip_in_place(
            flipping_up=flipping_up,
            flipping_index=flipping_index,
        )

        original_state_psi = before_swap_system_state.get_Psi_of_N()
        proposed_state_psi = after_swap_system_state.get_Psi_of_N()
        psi_factor = float(
            np.real(proposed_state_psi * np.conjugate(proposed_state_psi))
            / np.real(original_state_psi * np.conjugate(original_state_psi))
        )

        # return un-optimized default
        return (
            self.get_H_eff_difference(
                time=time,
                system_state_a=before_swap_system_state,
                system_state_b=after_swap_system_state,
            ),
            psi_factor,
        )

    def get_exp_H_eff(
        self,
        time: float,
        system_state: state.SystemState,
    ) -> float:
        return np.exp(self.get_H_eff(system_state=system_state, time=time))

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return {
            "U": self.U,
            "E": self.E,
            "J": self.J,
            "phi": self.phi,
            **additional_info,
        }


class VPartsMapping(Enum):
    A = "a"
    B = "b"
    C = "c"


class HardcoreBosonicHamiltonianStraightCalcPsiDiff(Hamiltonian):
    """This is implemented EXTREMELY inefficient.
    This used to make more sense in the past, when there would be 3*8 operator-strings with different factors that later would have been required to be re-assembled.
    In this form it is ONLY used for checking later and more streamlined implementations for correctness.
    """

    def __init__(
        self,
        U: float,
        E: float,
        J: float,
        phi: float,
    ):
        super().__init__(U=U, E=E, J=J, phi=phi)

    def get_base_energy(
        self,
        system_state: state.SystemState,
    ) -> float:
        u_count = 0.0
        eps_collector = 0.0

        for index in range(system_state.get_number_sites_wo_spin_degree()):
            # opposite spin
            index_os = system_state.get_opposite_spin_index(index)

            # count number of double occupancies
            u_count += (
                system_state.get_state_array()[index]
                * system_state.get_state_array()[index_os]
            )

            # collect epsilon values
            eps_collector += system_state.get_state_array()[
                index
            ] * system_state.get_eps_multiplier(
                index=index, phi=self.phi, sin_phi=self.sin_phi, cos_phi=self.cos_phi
            ) + system_state.get_state_array()[
                index_os
            ] * system_state.get_eps_multiplier(
                index=index_os, phi=self.phi, sin_phi=self.sin_phi, cos_phi=self.cos_phi
            )

        return self.U * u_count + self.E * eps_collector

    def get_H_n(
        self,
        time: float,
        system_state: state.SystemState,
    ) -> np.complex128:
        operator_evaluations = self.V_parts(system_state=system_state)

        psi_N = system_state.get_Psi_of_N()

        total_sum = np.complex128(0)

        for val in VPartsMapping:
            for l, m, K in operator_evaluations[val]:
                psi_K = K.get_Psi_of_N()

                eps_m_minus_eps_l = self.E * (
                    system_state.get_eps_multiplier(
                        index=m,
                        phi=self.phi,
                        sin_phi=self.sin_phi,
                        cos_phi=self.cos_phi,
                    )
                    - system_state.get_eps_multiplier(
                        index=l,
                        phi=self.phi,
                        sin_phi=self.sin_phi,
                        cos_phi=self.cos_phi,
                    )
                )

                # probably inefficient and is always 1 but wanted to do it properly
                # properly configured mc sampling doesn't touch this function and assumes homogenous psi_N, so this is not major efficiency problem
                psi_K_over_psi_N = psi_K / psi_N

                product = np.complex128(1)
                product *= psi_K_over_psi_N

                if val == VPartsMapping.A:
                    # A part of the first order
                    one_over_epsm_minus_epsl = 1 / eps_m_minus_eps_l
                    e_to_the_t_epsm_minus_epsl_minus_one = np.expm1(
                        1j * time * eps_m_minus_eps_l
                    )
                    product *= (
                        one_over_epsm_minus_epsl * e_to_the_t_epsm_minus_epsl_minus_one
                    )
                elif val == VPartsMapping.B:
                    # B part of the first order
                    one_over_epsm_minus_epsl_plus_U = 1 / (eps_m_minus_eps_l + self.U)
                    e_to_the_t_epsm_minus_epsl_plus_U_minus_one = np.expm1(
                        1j * time * (eps_m_minus_eps_l + self.U)
                    )
                    product *= (
                        one_over_epsm_minus_epsl_plus_U
                        * e_to_the_t_epsm_minus_epsl_plus_U_minus_one
                    )
                elif val == VPartsMapping.C:
                    # C part of the first order
                    one_over_epsm_minus_epsl_minus_U = 1 / (eps_m_minus_eps_l - self.U)
                    e_to_the_t_epsm_minus_epsl_minus_U_minus_one = np.expm1(
                        1j * time * (eps_m_minus_eps_l - self.U)
                    )
                    product *= (
                        one_over_epsm_minus_epsl_minus_U
                        * e_to_the_t_epsm_minus_epsl_minus_U_minus_one
                    )

                total_sum += product

        return total_sum * self.J

    def V_parts(
        self,
        system_state: state.SystemState,
    ) -> Dict[VPartsMapping, List[Tuple[int, int, int, state.SystemState]]]:
        """
        returns Tuple[l,m,K] The neighbor summation indices l&m and the resulting state K where a match was made
        """
        number_sites = system_state.get_number_sites_wo_spin_degree()

        result: Dict[VPartsMapping, List[Tuple[int, int, state.SystemState]]] = {}
        for val in VPartsMapping:
            result[val] = []

        # CAUTION we NEED doubling here. Meaning for (l=0,m=1) we WANT (l=1,m=0) ALSO
        # This resulted in a notation hickup in Calculate_V_5_rename_merge.mmdata that was corrected for in 01-EquationsOfMotionForOperators.xopp

        for l in range(number_sites):
            l_os = system_state.get_opposite_spin_index(l)

            index_neighbors = system_state.get_nearest_neighbor_indices(l)
            index_os_neighbors = system_state.get_nearest_neighbor_indices(l_os)

            for m, m_os in zip(index_neighbors, index_os_neighbors, strict=True):
                # The operators act left onto <system_state_array|operator|output K>

                m_to_l_hopped_state = None

                def get_m_to_l_hopped_state_array() -> state.SystemState:
                    nonlocal m_to_l_hopped_state
                    if m_to_l_hopped_state is None:
                        m_to_l_hopped_state = system_state.get_editable_copy()
                        m_to_l_hopped_state.get_state_array()[l] = 1
                        m_to_l_hopped_state.get_state_array()[m] = 0
                    return m_to_l_hopped_state

                os_m_to_l_hopped_state = None

                def get_os_m_to_l_hopped_state_array() -> state.SystemState:
                    nonlocal os_m_to_l_hopped_state
                    if os_m_to_l_hopped_state is None:
                        os_m_to_l_hopped_state = system_state.get_editable_copy()
                        os_m_to_l_hopped_state.get_state_array()[l_os] = 1
                        os_m_to_l_hopped_state.get_state_array()[m_os] = 0
                    return os_m_to_l_hopped_state

                system_state_array = system_state.get_state_array()

                # A: c_l * c#_m (1 + 2 nd_l nd_m - nd_l - nd_m)
                if (
                    (system_state_array[l] == 0)
                    and (system_state_array[m] == 1)
                    and (
                        1
                        + 2
                        * (system_state_array[l_os] == 1)
                        * (system_state_array[m_os] == 1)
                        - (system_state_array[m_os] == 1)
                        - (system_state_array[l_os] == 1)
                    )
                ):
                    result[VPartsMapping.A].append(
                        (l, m, get_m_to_l_hopped_state_array())
                    )

                # A: d_l * d#_m (1 + 2 nc_l nc_m - nc_l - nc_m)
                if (
                    (system_state_array[l_os] == 0)
                    and (system_state_array[m_os] == 1)
                    and (
                        1
                        + 2
                        * (system_state_array[l] == 1)
                        * (system_state_array[m] == 1)
                        - (system_state_array[m] == 1)
                        - (system_state_array[l] == 1)
                    )
                ):
                    result[VPartsMapping.A].append(
                        (l, m, get_os_m_to_l_hopped_state_array())
                    )

                # B: c_l * c#_m * nd_m (1 - nd_l)
                if (
                    (system_state_array[l] == 0)
                    and (system_state_array[m] == 1)
                    and (system_state_array[m_os] == 1)
                    and (1 - (system_state_array[l_os] == 1))
                ):
                    result[VPartsMapping.B].append(
                        (l, m, get_m_to_l_hopped_state_array())
                    )

                # B: d_l * d#_m * nc_m (1 - nc_l)
                if (
                    (system_state_array[l_os] == 0)
                    and (system_state_array[m_os] == 1)
                    and (system_state_array[m] == 1)
                    and (1 - (system_state_array[l] == 1))
                ):
                    result[VPartsMapping.B].append(
                        (l, m, get_os_m_to_l_hopped_state_array())
                    )

                # C: c_l * c#_m * nd_l (1 - nd_m)
                if (
                    (system_state_array[l] == 0)
                    and (system_state_array[m] == 1)
                    and (system_state_array[l_os] == 1)
                    and (1 - (system_state_array[m_os] == 1))
                ):
                    result[VPartsMapping.C].append(
                        (l, m, get_m_to_l_hopped_state_array())
                    )

                # B: d_l * d#_m * nc_l (1 - nc_m)
                if (
                    (system_state_array[l_os] == 0)
                    and (system_state_array[m_os] == 1)
                    and (system_state_array[l] == 1)
                    and (1 - (system_state_array[m] == 1))
                ):
                    result[VPartsMapping.C].append(
                        (l, m, get_os_m_to_l_hopped_state_array())
                    )

        return result

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return super().get_log_info(
            {
                "type": "HardcoreBosonicHamiltonianStraightCalcPsiDiff",
                **additional_info,
            }
        )


class HardcoreBosonicHamiltonian(Hamiltonian):
    def __init__(
        self,
        U: float,
        E: float,
        J: float,
        phi: float,
        initial_system_state: state.InitialSystemState,
    ):
        super().__init__(U=U, E=E, J=J, phi=phi)

        if not isinstance(initial_system_state, state.HomogenousInitialSystemState):
            raise Exception(
                "The simplified Hamiltonian requires a HomogenousInitialSystemState as a pre-requirement"
            )

    def get_base_energy(
        self,
        system_state: state.SystemState,
    ) -> float:
        u_count = 0.0
        eps_collector = 0.0

        for index in range(system_state.get_number_sites_wo_spin_degree()):
            # opposite spin
            index_os = system_state.get_opposite_spin_index(index)

            # count number of double occupancies
            u_count += (
                system_state.get_state_array()[index]
                * system_state.get_state_array()[index_os]
            )

            # collect epsilon values
            eps_collector += system_state.get_state_array()[
                index
            ] * system_state.get_eps_multiplier(
                index=index, phi=self.phi, sin_phi=self.sin_phi, cos_phi=self.cos_phi
            ) + system_state.get_state_array()[
                index_os
            ] * system_state.get_eps_multiplier(
                index=index_os, phi=self.phi, sin_phi=self.sin_phi, cos_phi=self.cos_phi
            )

        return self.U * u_count + self.E * eps_collector

    def get_H_n(
        self,
        time: float,
        system_state: state.SystemState,
    ) -> np.complex128:
        number_sites = system_state.get_number_sites_wo_spin_degree()

        total_sum: np.complex128 = np.complex128(0)

        # CAUTION we NEED doubling here. Meaning for (l=0,m=1) we WANT (l=1,m=0) ALSO

        for l in range(number_sites):
            l_os = system_state.get_opposite_spin_index(l)

            index_neighbors = system_state.get_nearest_neighbor_indices(l)

            neighbors_occupation_tuples = [
                (
                    self.E
                    * system_state.get_eps_multiplier(
                        index=nb,
                        phi=self.phi,
                        sin_phi=self.sin_phi,
                        cos_phi=self.cos_phi,
                    ),
                    system_state.get_state_array()[nb],
                    system_state.get_state_array()[
                        system_state.get_opposite_spin_index(nb)
                    ],
                )
                for nb in index_neighbors
            ]

            total_sum += calculate_v_plain(
                U=self.U,
                t=time,
                epsl=self.E
                * system_state.get_eps_multiplier(
                    index=l, phi=self.phi, sin_phi=self.sin_phi, cos_phi=self.cos_phi
                ),
                occ_l_up=system_state.get_state_array()[l],
                occ_l_down=system_state.get_state_array()[l_os],
                neighbors_eps_occupation_tuples=neighbors_occupation_tuples,
            )

        return total_sum * self.J

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return super().get_log_info(
            {
                "type": "HardcoreBosonicHamiltonian",
                **additional_info,
            }
        )


class HardcoreBosonicHamiltonianSwappingOptimization(HardcoreBosonicHamiltonian):
    def __init__(
        self,
        U: float,
        E: float,
        J: float,
        phi: float,
        initial_system_state: state.InitialSystemState,
    ):
        super().__init__(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )

        if not isinstance(initial_system_state, state.HomogenousInitialSystemState):
            raise Exception(
                "The simplified Hamiltonian requires a HomogenousInitialSystemState as a pre-requirement"
            )

    def get_base_energy_difference_swapping(
        self,
        sw1_up: bool,
        sw1_index: int,
        sw1_occupation: int,
        sw1_occupation_os: int,
        sw2_up: bool,
        sw2_index: int,
        sw2_occupation: int,
        sw2_occupation_os: int,
        before_swap_system_state: state.SystemState,
    ) -> float:
        res = 0

        # double occupations

        if sw1_up == sw2_up:
            res += (
                self.U
                * (sw1_occupation - sw2_occupation)
                * (sw1_occupation_os - sw2_occupation_os)
            )
        if sw1_up != sw2_up:
            res += (
                self.U
                * (sw1_occupation - sw2_occupation_os)
                * (sw1_occupation_os - sw2_occupation)
            )

        # electrical field
        eps_i = before_swap_system_state.get_eps_multiplier(
            index=sw1_index, phi=self.phi, sin_phi=self.sin_phi, cos_phi=self.cos_phi
        )
        eps_j = before_swap_system_state.get_eps_multiplier(
            index=sw2_index, phi=self.phi, sin_phi=self.sin_phi, cos_phi=self.cos_phi
        )
        i_occupation = sw1_occupation
        if not sw1_up:
            i_occupation = sw1_occupation_os
        j_occupation = sw2_occupation
        if not sw2_up:
            j_occupation = sw2_occupation_os

        res += self.E * (eps_i - eps_j) * (i_occupation - j_occupation)

        return res

    def get_H_eff_difference_swapping(
        self,
        time: float,
        sw1_up: bool,
        sw1_index: int,
        sw2_up: bool,
        sw2_index: int,
        before_swap_system_state: state.SystemState,
    ) -> Tuple[np.complex128, float]:
        sw1_occupation = before_swap_system_state.get_state_array()[sw1_index]
        sw1_occupation_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(sw1_index)
        ]
        sw2_occupation = before_swap_system_state.get_state_array()[sw2_index]
        sw2_occupation_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(sw2_index)
        ]

        if sw1_occupation * sw1_up + sw1_occupation_os * (
            not sw1_up
        ) == sw2_occupation * sw2_up + sw2_occupation_os * (not sw2_up):
            # The swapped occupations are equal. We know the result
            return (np.complex128(0), 1.0)

        sw1_eps = before_swap_system_state.get_eps_multiplier(
            index=sw1_index,
            phi=self.phi,
            sin_phi=self.sin_phi,
            cos_phi=self.cos_phi,
        )
        sw2_eps = before_swap_system_state.get_eps_multiplier(
            index=sw2_index,
            phi=self.phi,
            sin_phi=self.sin_phi,
            cos_phi=self.cos_phi,
        )
        sw1_neighbor_eps_occupation_direct_tuples = [
            (
                before_swap_system_state.get_eps_multiplier(
                    index=nb,
                    phi=self.phi,
                    sin_phi=self.sin_phi,
                    cos_phi=self.cos_phi,
                ),
                before_swap_system_state.get_state_array()[nb],
                before_swap_system_state.get_state_array()[
                    before_swap_system_state.get_opposite_spin_index(nb)
                ],
                nb == sw2_index,
            )
            for nb in before_swap_system_state.get_nearest_neighbor_indices(sw1_index)
        ]
        sw2_neighbor_eps_occupation_direct_tuples = [
            (
                before_swap_system_state.get_eps_multiplier(
                    index=nb,
                    phi=self.phi,
                    sin_phi=self.sin_phi,
                    cos_phi=self.cos_phi,
                ),
                before_swap_system_state.get_state_array()[nb],
                before_swap_system_state.get_state_array()[
                    before_swap_system_state.get_opposite_spin_index(nb)
                ],
                nb == sw1_index,
            )
            for nb in before_swap_system_state.get_nearest_neighbor_indices(sw2_index)
        ]

        unscaled_H_n_difference = np.complex128(0)

        unscaled_H_n_difference += calculate_v_hop(
            hop_sw1_up=sw1_up,
            hop_sw2_up=sw2_up,
            U=self.U,
            t=time,
            eps_sw1=sw1_eps,
            occ_sw1_up=sw1_occupation,
            occ_sw1_down=sw1_occupation_os,
            occ_sw2_up=sw2_occupation,
            occ_sw2_down=sw2_occupation_os,
            neighbors_eps_occupation_tuples=sw1_neighbor_eps_occupation_direct_tuples,
        )
        unscaled_H_n_difference += calculate_v_hop(
            hop_sw1_up=sw2_up,
            hop_sw2_up=sw1_up,
            U=self.U,
            t=time,
            eps_sw1=sw2_eps,
            occ_sw1_up=sw2_occupation,
            occ_sw1_down=sw2_occupation_os,
            occ_sw2_up=sw1_occupation,
            occ_sw2_down=sw1_occupation_os,
            neighbors_eps_occupation_tuples=sw2_neighbor_eps_occupation_direct_tuples,
        )

        for (
            neighbor_as_center_eps,
            neighbor_as_center_occ,
            neighbor_as_center_occ_os,
            same,
        ) in sw1_neighbor_eps_occupation_direct_tuples:
            singular_important_neighbor = [
                (
                    sw1_eps,
                    sw1_occupation,
                    sw1_occupation_os,
                    same,
                )
            ]
            unscaled_H_n_difference += calculate_v_hop(
                hop_sw1_up=sw1_up,
                hop_sw2_up=sw2_up,
                U=self.U,
                t=time,
                eps_sw1=neighbor_as_center_eps,
                occ_sw1_up=neighbor_as_center_occ,
                occ_sw1_down=neighbor_as_center_occ_os,
                occ_sw2_up=sw2_occupation,
                occ_sw2_down=sw2_occupation_os,
                neighbors_eps_occupation_tuples=singular_important_neighbor,
            )

        for (
            neighbor_as_center_eps,
            neighbor_as_center_occ,
            neighbor_as_center_occ_os,
            same,
        ) in sw2_neighbor_eps_occupation_direct_tuples:
            singular_important_neighbor = [
                (
                    sw2_eps,
                    sw2_occupation,
                    sw2_occupation_os,
                    same,
                )
            ]
            unscaled_H_n_difference += calculate_v_hop(
                hop_sw1_up=sw2_up,
                hop_sw2_up=sw1_up,
                U=self.U,
                t=time,
                eps_sw1=neighbor_as_center_eps,
                occ_sw1_up=neighbor_as_center_occ,
                occ_sw1_down=neighbor_as_center_occ_os,
                occ_sw2_up=sw1_occupation,
                occ_sw2_down=sw1_occupation_os,
                neighbors_eps_occupation_tuples=singular_important_neighbor,
            )

        return (
            self.J * unscaled_H_n_difference
            - (
                1j
                * self.get_base_energy_difference_swapping(
                    sw1_up=sw1_up,
                    sw1_index=sw1_index,
                    sw1_occupation=sw1_occupation,
                    sw1_occupation_os=sw1_occupation_os,
                    sw2_up=sw2_up,
                    sw2_index=sw2_index,
                    sw2_occupation=sw2_occupation,
                    sw2_occupation_os=sw2_occupation_os,
                    before_swap_system_state=before_swap_system_state,
                )
                * time
            ),
            1.0,  # this being 1.0 is a required assumption for this simplification
        )

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return super().get_log_info(
            {
                "type": "HardcoreBosonicHamiltonianSwappingOptimization",
                **additional_info,
            }
        )


class HardcoreBosonicHamiltonianFlippingOptimization(HardcoreBosonicHamiltonian):
    def __init__(
        self,
        U: float,
        E: float,
        J: float,
        phi: float,
        initial_system_state: state.InitialSystemState,
    ):
        super().__init__(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )

        if not isinstance(initial_system_state, state.HomogenousInitialSystemState):
            raise Exception(
                "The simplified Hamiltonian requires a HomogenousInitialSystemState as a pre-requirement"
            )

    def get_base_energy_difference_flipping(
        self,
        flipping_up: bool,
        flipping_index: int,
        flipping_occupation_before_flip: int,
        flipping_occupation_before_flip_os: int,
        before_swap_system_state: state.SystemState,
    ) -> float:
        res = 0

        # double occupations
        if flipping_up:
            res += self.U * (
                flipping_occupation_before_flip_os
                * (2 * flipping_occupation_before_flip - 1)
            )
        else:
            res += self.U * (
                flipping_occupation_before_flip
                * (2 * flipping_occupation_before_flip_os - 1)
            )

        # electrical field
        eps_i = before_swap_system_state.get_eps_multiplier(
            index=flipping_index,
            phi=self.phi,
            sin_phi=self.sin_phi,
            cos_phi=self.cos_phi,
        )
        i_occupation = flipping_occupation_before_flip
        if not flipping_up:
            i_occupation = flipping_occupation_before_flip_os
        res += self.E * eps_i * (2 * i_occupation - 1)

        return res

    def get_H_eff_difference_flipping(
        self,
        time: float,
        flipping_up: bool,
        flipping_index: int,
        before_swap_system_state: state.SystemState,
    ) -> Tuple[np.complex128, float]:
        flipping_occupation_before_flip = before_swap_system_state.get_state_array()[
            flipping_index
        ]
        flipping_occupation_before_flip_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(flipping_index)
        ]
        flipping_eps = self.E * before_swap_system_state.get_eps_multiplier(
            index=flipping_index,
            phi=self.phi,
            sin_phi=self.sin_phi,
            cos_phi=self.cos_phi,
        )

        flipping_neighbor_indices = (
            before_swap_system_state.get_nearest_neighbor_indices(flipping_index)
        )
        flipping_neighbors_eps_occupation_tuples = [
            (
                self.E
                * before_swap_system_state.get_eps_multiplier(
                    index=nb,
                    phi=self.phi,
                    sin_phi=self.sin_phi,
                    cos_phi=self.cos_phi,
                ),
                before_swap_system_state.get_state_array()[nb],
                before_swap_system_state.get_state_array()[
                    before_swap_system_state.get_opposite_spin_index(nb)
                ],
            )
            for nb in flipping_neighbor_indices
        ]

        # treat all cases with l flipped
        unscaled_H_n_difference = calculate_v_flip(
            flip_up=flipping_up,
            U=self.U,
            t=time,
            epsl=flipping_eps,
            occ_l_up=flipping_occupation_before_flip,
            occ_l_down=flipping_occupation_before_flip_os,
            neighbors_eps_occupation_tuples=flipping_neighbors_eps_occupation_tuples,
        )

        return (
            self.J * unscaled_H_n_difference
            - (
                1j
                * self.get_base_energy_difference_flipping(
                    flipping_up=flipping_up,
                    flipping_index=flipping_index,
                    flipping_occupation_before_flip=flipping_occupation_before_flip,
                    flipping_occupation_before_flip_os=flipping_occupation_before_flip_os,
                    before_swap_system_state=before_swap_system_state,
                )
                * time
            ),
            1.0,  # this being 1.0 is a required assumption for this simplification
        )

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return super().get_log_info(
            {
                "type": "HardcoreBosonicHamiltonianFlippingOptimization",
                **additional_info,
            }
        )


# TODO python CAN do multiple inheritance https://www.programiz.com/python-programming/multiple-inheritance
class HardcoreBosonicHamiltonianFlippingAndSwappingOptimization(
    HardcoreBosonicHamiltonianFlippingOptimization
):
    def __init__(
        self,
        U: float,
        E: float,
        J: float,
        phi: float,
        initial_system_state: state.InitialSystemState,
    ):
        super().__init__(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )

        if not isinstance(initial_system_state, state.HomogenousInitialSystemState):
            raise Exception(
                "The simplified Hamiltonian requires a HomogenousInitialSystemState as a pre-requirement"
            )

        self.swapping_hamiltonian = HardcoreBosonicHamiltonianSwappingOptimization(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )

    # delegate this call, as we cannot extend multiple classes
    def get_H_eff_difference_swapping(
        self,
        time: float,
        sw1_up: bool,
        sw1_index: int,
        sw2_up: bool,
        sw2_index: int,
        before_swap_system_state: state.SystemState,
    ) -> Tuple[np.complex128, float]:
        return self.swapping_hamiltonian.get_H_eff_difference_swapping(
            time=time,
            sw1_up=sw1_up,
            sw1_index=sw1_index,
            sw2_up=sw2_up,
            sw2_index=sw2_index,
            before_swap_system_state=before_swap_system_state,
        )

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return super().get_log_info(
            {
                "type": "HardcoreBosonicHamiltonianFlippingAndSwappingOptimization",
                **additional_info,
            }
        )
