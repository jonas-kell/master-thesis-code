from typing import Dict, List, Tuple, Union, Any, TYPE_CHECKING
from enum import Enum
from abc import ABC, abstractmethod
import state
import numpy as np
from vcomponents import (
    v as calculate_v_plain,
    v_flip as calculate_v_flip,
    v_hop as calculate_v_hop,
    v_double_flip as calculate_v_double_flip,
)

if TYPE_CHECKING:
    # WTF python https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
    import sampler


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
        if sw1_index == sw2_index:
            # assume swapping on the same site is forbidden, because that edge case is not handled in the mathematics properly
            raise Exception("Not allowed to request swapping from and to the same site")

        sw1_occupation = before_swap_system_state.get_state_array()[sw1_index]
        sw1_occupation_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(sw1_index)
        ]
        sw2_occupation = before_swap_system_state.get_state_array()[sw2_index]
        sw2_occupation_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(sw2_index)
        ]

        # do it here in more easy to read but more expensive logic, as this is not used, but the optimaizations
        if (
            (sw1_up and sw2_up and sw1_occupation == sw2_occupation)
            or (not sw1_up and sw2_up and sw1_occupation_os == sw2_occupation)
            or (sw1_up and not sw2_up and sw1_occupation == sw2_occupation_os)
            or (not sw1_up and not sw2_up and sw1_occupation_os == sw2_occupation_os)
        ):
            # The swapped occupations are equal. We know the result
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

    def get_H_eff_difference_double_flipping(
        self,
        time: float,
        flipping1_up: bool,
        flipping1_index: int,
        flipping2_up: bool,
        flipping2_index: int,
        before_swap_system_state: state.SystemState,  # required, because un-optimized implementation uses state and optimized implementation uses it to pre-compute the occupations and lambda functions
    ) -> Tuple[np.complex128, float]:
        # allocate swapped state
        after_swap_system_state = before_swap_system_state.get_editable_copy()
        after_swap_system_state.flip_in_place(
            flipping_up=flipping1_up,
            flipping_index=flipping1_index,
        )
        after_swap_system_state.flip_in_place(
            flipping_up=flipping2_up,
            flipping_index=flipping2_index,
        )

        if flipping1_index == flipping2_index:
            # also trigger this if the spin directions are not the same.
            # That would make sense, but external mathematics performed on this do not take on-site possibilities into consideration
            raise Exception(
                "Not allowed to request double flipping on the same site as it is not clear what this means"
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


class HardcoreBosonicHamiltonianExact(Hamiltonian):
    """This is requires diagonalization, so it will have exponential time-complexity!!
    Only check for correctness on small systems with this.
    """

    def __init__(
        self,
        U: float,
        E: float,
        J: float,
        phi: float,
        system_geometry: state.SystemGeometry,
        exact_sampler: "sampler.ExactSampler",
    ):
        super().__init__(U=U, E=E, J=J, phi=phi)

        # construct the hamiltonian matrix
        self.base_states = []
        for a_base_state in exact_sampler.sample_generator(
            time=0,  # that doesn't matter for exact sampler
            random_generator=None,  # that doesn't matter for exact sampler
            worker_index=0,  # as only one worker in constructor
            num_workers=1,  # as only one worker in constructor
        ):
            self.base_states.append(a_base_state.get_state_array().copy())

        self.index_map = {tuple(entry): i for i, entry in enumerate(self.base_states)}

        def compare_bra_ket(bra, ket):
            return np.all(bra == ket)

        # does c#_l c_m
        def compare_m_to_l_hopped_ket_to_bra(bra, ket, l, m):
            if ket[m] == 0:
                return False
            if ket[l] == 1:
                return False

            copy_of_ket = np.copy(ket)

            copy_of_ket[m] = 0
            copy_of_ket[l] = 1

            return compare_bra_ket(bra, copy_of_ket)

        print("Exact Hamiltonian is done generating basis and lookup")
        self.H = np.array(
            [
                [
                    +self.U
                    * np.sum(
                        np.array(
                            [
                                (
                                    1
                                    * compare_bra_ket(bra_state, ket_state)
                                    * bra_state[index]
                                    * bra_state[
                                        system_geometry.get_opposite_spin_index(index)
                                    ]
                                )
                                for index in range(
                                    system_geometry.get_number_sites_wo_spin_degree()
                                )
                            ]
                        )
                    )
                    + self.E
                    * np.sum(
                        np.array(
                            [
                                (
                                    1
                                    * system_geometry.get_eps_multiplier(
                                        index=index,
                                        phi=self.phi,
                                        cos_phi=self.cos_phi,
                                        sin_phi=self.sin_phi,
                                    )
                                    * compare_bra_ket(bra_state, ket_state)
                                    * bra_state[index]
                                )
                                for index in range(system_geometry.get_number_sites())
                            ]
                        )
                    )
                    - J
                    * np.sum(
                        np.array(
                            [
                                (
                                    1
                                    * compare_m_to_l_hopped_ket_to_bra(
                                        bra_state, ket_state, ind1, ind2
                                    )
                                )
                                for (ind1, ind2) in (
                                    [
                                        (index, neighbor_index)
                                        for index in range(
                                            system_geometry.get_number_sites()
                                        )
                                        for neighbor_index in system_geometry.get_nearest_neighbor_indices(
                                            index=index
                                        )
                                    ]
                                )
                            ]
                        )
                    )
                    for ket_state in self.base_states
                ]
                for bra_state in self.base_states
            ]
        )
        print("Exact Hamiltonian is done generating Hamiltonian")
        if not np.all(np.conjugate(self.H.T) == self.H):
            print("H not hermetian")

        self.d = 2 ** (system_geometry.get_number_sites())
        # Dimension of the Hilbert space 2 spin degrees on n particles
        amplitude = 1 / np.sqrt(self.d)  # Same amplitude for all basis states
        self.psi_0 = np.full(self.d, amplitude, dtype=complex)

        self.matrix_cache = None
        self.matrix_cache_time: float = -1123123123123123123
        self.recalculate_matrix(time=0)

    def recalculate_matrix(self, time: float):
        # dependency only needed when unsing this hamiltonian
        from scipy.linalg import expm

        if self.matrix_cache_time == time:
            return  # is already cached
        else:
            # recalc the cache
            self.matrix_cache = expm(-1j * self.H * time)

            self.matrix_cache_time = time

    def get_one_hot_vector(self, basis_elem: np.ndarray) -> np.ndarray:
        vec = np.zeros(self.d, dtype=np.complex128)
        index = self.index_map[tuple(basis_elem)]
        vec[index] = 1

        return vec

    def get_base_energy(
        self,
        system_state: state.SystemState,
    ) -> float:
        _ = system_state

        raise Exception(
            "This should not be called, as we do not support accessing the base energy directly on the exact version"
        )

    def get_H_n(
        self,
        time: float,
        system_state: state.SystemState,
    ) -> np.complex128:
        _ = system_state
        _ = time

        raise Exception(
            "This should not be called, as we do not support accessing the H_n directly on the exact version"
        )

    def get_H_eff(
        self,
        time: float,
        system_state: state.SystemState,
    ) -> np.complex128:
        self.recalculate_matrix(time=time)

        return np.log(
            np.vdot(
                self.get_one_hot_vector(system_state.get_state_array()),
                np.dot(self.matrix_cache, self.psi_0),
            )
        )

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return super().get_log_info(
            {
                "type": "HardcoreBosonicHamiltonianExact",
                **additional_info,
            }
        )


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
        if sw1_index == sw2_index:
            # assume swapping on the same site is forbidden, because that edge case is not handled in the mathematics properly
            raise Exception("Not allowed to request swapping from and to the same site")

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

        sw1_eps = self.E * before_swap_system_state.get_eps_multiplier(
            index=sw1_index,
            phi=self.phi,
            sin_phi=self.sin_phi,
            cos_phi=self.cos_phi,
        )
        sw2_eps = self.E * before_swap_system_state.get_eps_multiplier(
            index=sw2_index,
            phi=self.phi,
            sin_phi=self.sin_phi,
            cos_phi=self.cos_phi,
        )
        sw1_neighbor_eps_occupation_direct_tuples = [
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
                nb == sw2_index,
            )
            for nb in before_swap_system_state.get_nearest_neighbor_indices(sw1_index)
        ]
        sw2_neighbor_eps_occupation_direct_tuples = [
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

    def get_H_eff_difference_double_flipping(
        self,
        time: float,
        flipping1_up: bool,
        flipping1_index: int,
        flipping2_up: bool,
        flipping2_index: int,
        before_swap_system_state: state.SystemState,
    ) -> Tuple[np.complex128, float]:
        if flipping1_index == flipping2_index:
            # also trigger this if the spin directions are not the same.
            # That would make sense, but external mathematics performed on this do not take on-site possibilities into consideration
            raise Exception(
                "Not allowed to request double flipping on the same site as it is not clear what this means"
            )

        flipping1_occupation_before_flip = before_swap_system_state.get_state_array()[
            flipping1_index
        ]
        flipping1_occupation_before_flip_os = (
            before_swap_system_state.get_state_array()[
                before_swap_system_state.get_opposite_spin_index(flipping1_index)
            ]
        )
        flipping1_eps = self.E * before_swap_system_state.get_eps_multiplier(
            index=flipping1_index,
            phi=self.phi,
            sin_phi=self.sin_phi,
            cos_phi=self.cos_phi,
        )
        flipping2_occupation_before_flip = before_swap_system_state.get_state_array()[
            flipping2_index
        ]
        flipping2_occupation_before_flip_os = (
            before_swap_system_state.get_state_array()[
                before_swap_system_state.get_opposite_spin_index(flipping2_index)
            ]
        )
        flipping2_eps = self.E * before_swap_system_state.get_eps_multiplier(
            index=flipping2_index,
            phi=self.phi,
            sin_phi=self.sin_phi,
            cos_phi=self.cos_phi,
        )

        flipping1_neighbor_indices = (
            before_swap_system_state.get_nearest_neighbor_indices(flipping1_index)
        )
        flipping1_neighbors_eps_occupation_direct_tuples = [
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
                nb == flipping2_index,
            )
            for nb in flipping1_neighbor_indices
        ]
        flipping2_neighbor_indices = (
            before_swap_system_state.get_nearest_neighbor_indices(flipping2_index)
        )
        flipping2_neighbors_eps_occupation_direct_tuples = [
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
                nb == flipping1_index,
            )
            for nb in flipping2_neighbor_indices
        ]

        unscaled_H_n_difference = np.complex128(0)

        unscaled_H_n_difference += calculate_v_double_flip(
            flip1_up=flipping1_up,
            flip2_up=flipping2_up,
            U=self.U,
            t=time,
            flip1_eps=flipping1_eps,
            flip1_occ_up=flipping1_occupation_before_flip,
            flip1_occ_down=flipping1_occupation_before_flip_os,
            neighbors_eps_occupation_tuples=flipping1_neighbors_eps_occupation_direct_tuples,
        )
        unscaled_H_n_difference += calculate_v_double_flip(
            flip1_up=flipping2_up,
            flip2_up=flipping1_up,
            U=self.U,
            t=time,
            flip1_eps=flipping2_eps,
            flip1_occ_up=flipping2_occupation_before_flip,
            flip1_occ_down=flipping2_occupation_before_flip_os,
            neighbors_eps_occupation_tuples=flipping2_neighbors_eps_occupation_direct_tuples,
        )

        return (
            self.J * unscaled_H_n_difference
            - (
                1j
                * self.get_base_energy_difference_double_flipping(
                    flipping1_up=flipping1_up,
                    flipping1_index=flipping1_index,
                    flipping1_occupation_before_flip=flipping1_occupation_before_flip,
                    flipping1_occupation_before_flip_os=flipping1_occupation_before_flip_os,
                    flipping2_up=flipping2_up,
                    flipping2_index=flipping2_index,
                    flipping2_occupation_before_flip=flipping2_occupation_before_flip,
                    flipping2_occupation_before_flip_os=flipping2_occupation_before_flip_os,
                    before_swap_system_state=before_swap_system_state,
                )
                * time
            ),
            1.0,  # this being 1.0 is a required assumption for this simplification
        )

    def get_base_energy_difference_double_flipping(
        self,
        flipping1_up: bool,
        flipping1_index: int,
        flipping1_occupation_before_flip: int,
        flipping1_occupation_before_flip_os: int,
        flipping2_up: bool,
        flipping2_index: int,
        flipping2_occupation_before_flip: int,
        flipping2_occupation_before_flip_os: int,
        before_swap_system_state: state.SystemState,
    ) -> float:
        if flipping1_index == flipping2_index:
            # also trigger this if the spin directions are not the same.
            # That would make sense, but external mathematics performed on this do not take on-site possibilities into consideration
            raise Exception(
                "Not allowed to request double flipping on the same site as it is not clear what this means"
            )
        res = 0

        # double occupations
        if flipping1_up:
            res += self.U * (
                flipping1_occupation_before_flip_os
                * (2 * flipping1_occupation_before_flip - 1)
            )
        else:
            res += self.U * (
                flipping1_occupation_before_flip
                * (2 * flipping1_occupation_before_flip_os - 1)
            )

        if flipping2_up:
            res += self.U * (
                flipping2_occupation_before_flip_os
                * (2 * flipping2_occupation_before_flip - 1)
            )
        else:
            res += self.U * (
                flipping2_occupation_before_flip
                * (2 * flipping2_occupation_before_flip_os - 1)
            )

        # electrical field
        eps_i1 = before_swap_system_state.get_eps_multiplier(
            index=flipping1_index,
            phi=self.phi,
            sin_phi=self.sin_phi,
            cos_phi=self.cos_phi,
        )
        i1_occupation = flipping1_occupation_before_flip
        if not flipping1_up:
            i1_occupation = flipping1_occupation_before_flip_os

        res += self.E * eps_i1 * (2 * i1_occupation - 1)
        eps_i2 = before_swap_system_state.get_eps_multiplier(
            index=flipping2_index,
            phi=self.phi,
            sin_phi=self.sin_phi,
            cos_phi=self.cos_phi,
        )
        i2_occupation = flipping2_occupation_before_flip
        if not flipping2_up:
            i2_occupation = flipping2_occupation_before_flip_os
        res += self.E * eps_i2 * (2 * i2_occupation - 1)

        return res

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
