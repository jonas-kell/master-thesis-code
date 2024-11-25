from typing import Dict, List, Tuple, Union, Any, TYPE_CHECKING, TypeAlias
import numpy.typing as npt
from enum import Enum
from abc import ABC, abstractmethod
import state
import systemgeometry
import numpy as np
from vcomponents import (
    v as calculate_v_plain,
    v_flip as calculate_v_flip,
    v_hop as calculate_v_hop,
    v_double_flip as calculate_v_double_flip,
)
from vcomponentssecondorder import v_second as v_second_order
from randomgenerator import RandomGenerator
from variationalclassicalnetworks import (
    PSISelection,
    ChainDirectionDependentAllSameFirstOrder,
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

    def initialize(
        self,
        time: float,
    ):
        _ = time

        # base hamiltonian does not need initializing

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

        # check that no unexpected too large indices are requested (this is typically not used, so no need for max efficiency)
        domain_size = before_swap_system_state.get_number_sites_wo_spin_degree()
        if (
            sw1_index < 0
            or sw2_index < 0
            or sw1_index >= domain_size
            or sw2_index >= domain_size
        ):
            raise Exception(
                f"Site must be bigger than 0 and smaller than {domain_size} to fit: {sw1_index} {sw2_index}"
            )

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
        # check that no unexpected too large indices are requested (this is typically not used, so no need for max efficiency)
        domain_size = before_swap_system_state.get_number_sites_wo_spin_degree()
        if flipping_index < 0 or flipping_index >= domain_size:
            raise Exception(
                f"Site must be bigger than 0 and smaller than {domain_size} to fit: {flipping_index}"
            )

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
        # check that no unexpected too large indices are requested (this is typically not used, so no need for max efficiency)
        domain_size = before_swap_system_state.get_number_sites_wo_spin_degree()
        if (
            flipping1_index < 0
            or flipping2_index < 0
            or flipping1_index >= domain_size
            or flipping2_index >= domain_size
        ):
            raise Exception(
                f"Site must be bigger than 0 and smaller than {domain_size} to fit: {flipping1_index} {flipping2_index}"
            )

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
    """
    This is requires diagonalization, so it will have exponential time-complexity!!
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
        number_of_workers: int,
    ):
        super().__init__(U=U, E=E, J=J, phi=phi)

        if number_of_workers != 1:
            # The expensive scipy/numpy operations will multithread anyway (seemed to do that in testing, when using hamiltonian=exact, sampler=exact)
            print(
                "Warning: Caching system will not handle more than one worker properly. For most efficient cpu usage, use hamiltonian=exact, sampler=exact, #workers=1"
            )

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
            print("H not hermitian (sanity check failed)")

        self.d = 2 ** (system_geometry.get_number_sites())
        # Dimension of the Hilbert space 2 spin degrees on n particles
        amplitude = 1 / np.sqrt(self.d)  # Same amplitude for all basis states
        self.psi_0 = np.full(self.d, amplitude, dtype=complex)

        self.matrix_cache = None
        self.matrix_cache_time: float = -1123123123123123123
        self.recalculate_matrix(time=0)

    def recalculate_matrix(self, time: float):
        # dependency only needed when unsing this hamiltonian (avoid requireing this on hpc-servers possibly)
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


class HardcoreBosonicHamiltonianStraightCalcPsiDiffFirstOrder(Hamiltonian):
    """
    This is implemented EXTREMELY inefficient.
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
                "type": "HardcoreBosonicHamiltonianStraightCalcPsiDiffFirstOrder",
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
            # index is correctly clamped to 0<=index<domain_size

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
            ] * system_state.get_eps_multiplier(  # eps multiplier explicitly allows for getting inputs > domain size
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
            # l is correctly clamped to 0<=index<domain_size
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
                    system_state.get_state_array()[  # nb is correctly clamped to 0<=index<domain_size
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


class HardcoreBosonicHamiltonianSecondOrder(HardcoreBosonicHamiltonian):
    def __init__(
        self,
        U: float,
        E: float,
        J: float,
        phi: float,
        initial_system_state: state.InitialSystemState,
        system_geometry: systemgeometry.SystemGeometry,
    ):
        if not isinstance(initial_system_state, state.HomogenousInitialSystemState):
            raise Exception(
                "The second order Hamiltonian requires a HomogenousInitialSystemState as a pre-requirement"
            )

        super().__init__(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )

        # second order requires this cache to be pre-calculated
        system_geometry.init_index_knows_cache(self.phi, self.sin_phi, self.cos_phi)
        self.system_geometry = system_geometry

    def get_H_n(
        self,
        time: float,
        system_state: state.SystemState,
    ) -> np.complex128:
        first_order_val = super().get_H_n(time=time, system_state=system_state)

        number_sites = system_state.get_number_sites_wo_spin_degree()

        second_order_total_sum: np.complex128 = np.complex128(0)

        for l in range(number_sites):
            # l is correctly clamped to 0<=index<domain_size
            input_tuples = self.system_geometry.get_index_knows_tuples(l)

            second_order_total_sum += v_second_order(
                U=self.U,
                E=self.E,
                t=time,
                knows_l_array=input_tuples,
                system_state=system_state,
            )

        return first_order_val - 0.5 * self.J * self.J * second_order_total_sum

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return super().get_log_info(
            {
                "type": "HardcoreBosonicHamiltonianSecondOrder",
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


# python COULD do multiple inheritance https://www.programiz.com/python-programming/multiple-inheritance
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
        return self.swapping_hamiltonian.get_base_energy_difference_swapping(
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


class ZerothOrderFlippingAndSwappingOptimization(
    HardcoreBosonicHamiltonianFlippingAndSwappingOptimization
):
    "Basically just sets H_n to 0, but has the swapping, flipping and double flipping optimizations for the base energy"

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

    def get_H_n(
        self,
        time: float,
        system_state: state.SystemState,
    ) -> np.complex128:
        _ = time
        _ = system_state

        return 0

    def get_H_eff_difference_swapping(
        self,
        time: float,
        sw1_up: bool,
        sw1_index: int,
        sw2_up: bool,
        sw2_index: int,
        before_swap_system_state: state.SystemState,
    ) -> Tuple[np.complex128, float]:
        occ1 = before_swap_system_state.get_state_array()[sw1_index]
        occ1_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(sw1_index)
        ]
        occ2 = before_swap_system_state.get_state_array()[sw2_index]
        occ2_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(sw2_index)
        ]

        return (
            -1j
            * time
            * self.get_base_energy_difference_swapping(
                sw1_up=sw1_up,
                sw1_index=sw1_index,
                sw1_occupation=occ1,
                sw1_occupation_os=occ1_os,
                sw2_up=sw2_up,
                sw2_index=sw2_index,
                sw2_occupation=occ2,
                sw2_occupation_os=occ2_os,
                before_swap_system_state=before_swap_system_state,
            ),
            1.0,
        )

    def get_H_eff_difference_flipping(
        self,
        time: float,
        flipping_up: bool,
        flipping_index: int,
        before_swap_system_state: state.SystemState,
    ) -> Tuple[np.complex128, float]:
        occ = before_swap_system_state.get_state_array()[flipping_index]
        occ_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(flipping_index)
        ]

        return (
            -1j
            * time
            * self.get_base_energy_difference_flipping(
                flipping_up=flipping_up,
                flipping_index=flipping_index,
                flipping_occupation_before_flip=occ,
                flipping_occupation_before_flip_os=occ_os,
                before_swap_system_state=before_swap_system_state,
            ),
            1.0,
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
        occ1 = before_swap_system_state.get_state_array()[flipping1_index]
        occ1_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(flipping1_index)
        ]
        occ2 = before_swap_system_state.get_state_array()[flipping2_index]
        occ2_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(flipping2_index)
        ]

        return (
            -1j
            * time
            * self.get_base_energy_difference_double_flipping(
                flipping1_up=flipping1_up,
                flipping1_index=flipping1_index,
                flipping1_occupation_before_flip=occ1,
                flipping1_occupation_before_flip_os=occ1_os,
                flipping2_up=flipping2_up,
                flipping2_index=flipping2_index,
                flipping2_occupation_before_flip=occ2,
                flipping2_occupation_before_flip_os=occ2_os,
                before_swap_system_state=before_swap_system_state,
            ),
            1.0,
        )

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return super().get_log_info(
            {
                "type": "ZerothOrderFlippingAndSwappingOptimization",
                **additional_info,
            }
        )


class ZerothOrder(HardcoreBosonicHamiltonian):
    "Basically just sets H_n to 0"

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

    def get_H_n(
        self,
        time: float,
        system_state: state.SystemState,
    ) -> np.complex128:
        _ = time
        _ = system_state

        return 0

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return super().get_log_info(
            {
                "type": "ZerothOrder",
                **additional_info,
            }
        )


class HardcoreBosonicHamiltonianFlippingAndSwappingOptimizationSecondOrder(
    HardcoreBosonicHamiltonianFlippingAndSwappingOptimization
):
    def __init__(
        self,
        U: float,
        E: float,
        J: float,
        phi: float,
        initial_system_state: state.InitialSystemState,
        system_geometry: systemgeometry.SystemGeometry,
    ):
        if not isinstance(initial_system_state, state.HomogenousInitialSystemState):
            raise Exception(
                "The second order Hamiltonian requires a HomogenousInitialSystemState as a pre-requirement"
            )

        super().__init__(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )

        self.second_order_base_hamiltonian = HardcoreBosonicHamiltonianSecondOrder(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            system_geometry=system_geometry,
        )
        # second order requires geometry cache to be pre-calculated -> will be done in constructor above
        self.system_geometry = system_geometry

    def get_H_n(
        self,
        time: float,
        system_state: state.SystemState,
    ) -> np.complex128:
        return self.second_order_base_hamiltonian.get_H_n(
            time=time, system_state=system_state
        )

    def get_H_eff_difference_swapping(
        self,
        time: float,
        sw1_up: bool,
        sw1_index: int,
        sw2_up: bool,
        sw2_index: int,
        before_swap_system_state: state.SystemState,
    ) -> Tuple[np.complex128, float]:
        use_sw1_index = sw1_index
        if not sw1_up:
            use_sw1_index = before_swap_system_state.get_opposite_spin_index(sw1_index)

        use_sw2_index = sw2_index
        if not sw2_up:
            use_sw2_index = before_swap_system_state.get_opposite_spin_index(sw2_index)

        if (
            before_swap_system_state.get_state_array()[use_sw1_index]
            == before_swap_system_state.get_state_array()[use_sw2_index]
        ):
            # The swapped occupations are equal. We know the result
            return (np.complex128(0), 1.0)

        first_order_val = super().get_H_eff_difference_swapping(
            time=time,
            sw1_up=sw1_up,
            sw1_index=sw1_index,
            sw2_up=sw2_up,
            sw2_index=sw2_index,
            before_swap_system_state=before_swap_system_state,
        )[0]

        filtered_index_knows_tuples = (
            self.system_geometry.get_index_knows_tuples_contains_two(
                sw1_index, sw2_index
            )
        )

        before = v_second_order(
            U=self.U,
            E=self.E,
            t=time,
            knows_l_array=filtered_index_knows_tuples,
            system_state=before_swap_system_state,
        )
        after = v_second_order(
            U=self.U,
            E=self.E,
            t=time,
            knows_l_array=filtered_index_knows_tuples,
            system_state=before_swap_system_state,
            flipping_tuples=[(sw1_index, sw1_up), (sw2_index, sw2_up)],
        )

        return (
            first_order_val - 0.5 * self.J * self.J * (before - after),
            1.0,  # this being 1.0 is a required assumption for this simplification
        )

    def get_H_eff_difference_flipping(
        self,
        time: float,
        flipping_up: bool,
        flipping_index: int,
        before_swap_system_state: state.SystemState,
    ) -> Tuple[np.complex128, float]:
        first_order_val = super().get_H_eff_difference_flipping(
            time=time,
            flipping_up=flipping_up,
            flipping_index=flipping_index,
            before_swap_system_state=before_swap_system_state,
        )[0]

        filtered_index_knows_tuples = (
            self.system_geometry.get_index_knows_tuples_contains_one(flipping_index)
        )

        before = v_second_order(
            U=self.U,
            E=self.E,
            t=time,
            knows_l_array=filtered_index_knows_tuples,
            system_state=before_swap_system_state,
        )
        after = v_second_order(
            U=self.U,
            E=self.E,
            t=time,
            knows_l_array=filtered_index_knows_tuples,
            system_state=before_swap_system_state,
            flipping_tuples=[(flipping_index, flipping_up)],
        )

        return (
            first_order_val - 0.5 * self.J * self.J * (before - after),
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
        first_order_val = super().get_H_eff_difference_double_flipping(
            time=time,
            flipping1_up=flipping1_up,
            flipping1_index=flipping1_index,
            flipping2_up=flipping2_up,
            flipping2_index=flipping2_index,
            before_swap_system_state=before_swap_system_state,
        )[0]

        filtered_index_knows_tuples = (
            self.system_geometry.get_index_knows_tuples_contains_two(
                flipping1_index, flipping2_index
            )
        )

        before = v_second_order(
            U=self.U,
            E=self.E,
            t=time,
            knows_l_array=filtered_index_knows_tuples,
            system_state=before_swap_system_state,
        )
        after = v_second_order(
            U=self.U,
            E=self.E,
            t=time,
            knows_l_array=filtered_index_knows_tuples,
            system_state=before_swap_system_state,
            flipping_tuples=[
                (flipping1_index, flipping1_up),
                (flipping2_index, flipping2_up),
            ],
        )

        return (
            first_order_val - 0.5 * self.J * self.J * (before - after),
            1.0,  # this being 1.0 is a required assumption for this simplification
        )

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return super().get_log_info(
            {
                "type": "HardcoreBosonicHamiltonianFlippingAndSwappingOptimizationSecondOrder",
                **additional_info,
            }
        )


ETAVecType: TypeAlias = npt.NDArray[np.complex128]


# We require the optimized difference and flipping cases for the base energy calculation
class VCNHardCoreBosonicHamiltonian(
    HardcoreBosonicHamiltonianFlippingAndSwappingOptimization
):
    eta_vec: ETAVecType  # typechecker whines around, when I use inline type annotation for this...

    def __init__(
        self,
        U: float,
        E: float,
        J: float,
        phi: float,
        initial_system_state: state.InitialSystemState,
        psi_selection: PSISelection,
        random_generator: RandomGenerator,
        init_sigma: float,
        eta_calculation_sampler: "sampler.GeneralSampler",
        pseudo_inverse_cutoff: float,
        variational_step_fraction_multiplier: int,
    ):
        if not isinstance(initial_system_state, state.HomogenousInitialSystemState):
            raise Exception(
                "The VCN Hamiltonian requires a HomogenousInitialSystemState as a pre-requirement"
            )

        super().__init__(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )

        self.psi_selection = psi_selection

        self.init_sigma = init_sigma

        self.eta_calculation_sampler = eta_calculation_sampler
        self.random_generator = random_generator.derive()

        self.variational_step_fraction_multiplier = variational_step_fraction_multiplier

        self.eta_vec = None

        self.current_time_cache = -12352343
        self.is_initializing = False
        self.pseudo_inverse_cutoff = pseudo_inverse_cutoff

    def get_base_energy_difference_l_to_m_hopping(
        self,
        l: int,
        m: int,
        spins_up: bool,
        before_swap_system_state: state.SystemState,
    ) -> float:
        domain_size = before_swap_system_state.get_number_sites_wo_spin_degree()

        use_l_index = l % domain_size
        use_m_index = m % domain_size

        l_occ = before_swap_system_state.get_state_array()[use_l_index]
        m_occ = before_swap_system_state.get_state_array()[use_m_index]
        l_occ_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(use_l_index)
        ]
        m_occ_os = before_swap_system_state.get_state_array()[
            before_swap_system_state.get_opposite_spin_index(use_m_index)
        ]

        if (spins_up and (l_occ != 1 or m_occ != 0)) or (
            not spins_up and (l_occ_os != 1 or m_occ_os != 0)
        ):
            # Either they are the same -> hopping does not change state, difference must be 0
            # OR because this is specifically l->m hopping test, that is a prefactor making it 0
            return 0

        return self.get_base_energy_difference_swapping(
            sw1_up=spins_up,
            sw1_index=use_l_index,
            sw1_occupation=l_occ,
            sw1_occupation_os=l_occ_os,
            sw2_up=spins_up,
            sw2_index=use_m_index,
            sw2_occupation=m_occ,
            sw2_occupation_os=m_occ_os,
            before_swap_system_state=before_swap_system_state,
        )

    def initialize(
        self,
        time: float,
    ):
        self.is_initializing = True

        if self.eta_vec is None:
            if time == 0:
                self.eta_vec = self.get_initialized_eta_vec()
                self.current_time_cache = 0
            else:
                raise Exception(
                    "The VCN Hamiltonian must start from a known set of eta params (t=0)"
                )

        num_etas = self.get_number_of_eta_parameters()

        prev_time = self.current_time_cache
        for intermediate_step_index in range(self.variational_step_fraction_multiplier):
            intermediate_step_time = (
                prev_time
                + (intermediate_step_index + 1)
                * (time - prev_time)
                / self.variational_step_fraction_multiplier
            )

            sample_generator_object = self.eta_calculation_sampler.sample_generator(
                time=intermediate_step_time,
                worker_index=0,
                num_workers=1,  # TODO training multithreaded
                random_generator=self.random_generator,
            )

            requires_probability_adjustment = (
                self.eta_calculation_sampler.requires_probability_adjustment()
            )

            normalization_factor = 0.0
            OO_averager = np.zeros(
                (
                    num_etas,
                    num_etas,
                ),
                dtype=np.complex128,
            )
            O_averager = np.zeros(
                (num_etas,),
                dtype=np.complex128,
            )
            EO_averager = np.zeros(
                (num_etas,),
                dtype=np.complex128,
            )
            E_averager = np.complex128(0)
            while True:
                try:
                    sampled_state_n = next(sample_generator_object)

                    if requires_probability_adjustment:
                        # sampled state needs to be scaled

                        # get_exp_H_eff is the most expensive calculation. Only do if absolutely necessary
                        h_eff = self.get_exp_H_eff(
                            time=intermediate_step_time, system_state=sampled_state_n
                        )
                        psi_n = sampled_state_n.get_Psi_of_N()

                        state_probability: float = np.real(np.conjugate(h_eff) * h_eff) * np.real(  # type: ignore -> this returns a scalar for sure
                            np.conjugate(psi_n) * psi_n
                        )
                    else:
                        # e.g. Monte Carlo. Normalization is only division by number of monte carlo samples
                        state_probability = 1.0

                    normalization_factor += state_probability

                    O_vector, E_loc = self.calculate_O_k_and_E_loc(
                        time=intermediate_step_time, system_state=sampled_state_n
                    )

                    O_vector_scaled = O_vector * state_probability
                    E_loc_scaled = E_loc * state_probability

                    OO_averager += np.outer(O_vector_scaled.conj(), O_vector_scaled)
                    O_averager += O_vector_scaled
                    EO_averager += E_loc_scaled * O_vector_scaled.conj()
                    E_averager += E_loc_scaled

                except StopIteration:
                    break

            O_averager_normed = O_averager / normalization_factor

            S_matrix = (OO_averager / normalization_factor) - np.outer(
                O_averager_normed.conj(), O_averager_normed
            )
            F_vector = (
                EO_averager / normalization_factor
                - (E_averager / normalization_factor) * O_averager_normed.conj()
            )

            pinv = np.linalg.pinv(
                S_matrix, hermitian=True, rcond=self.pseudo_inverse_cutoff
            )
            eta_derivative = -1j * (pinv @ F_vector)

            # now we have the derivative. Step with explicit euler integration # TODO better approximator
            self.eta_vec += eta_derivative

        # finished and the result is that trained self.eta_vec
        self.current_time_cache = time
        self.is_initializing = False

    def calculate_O_k_and_E_loc(
        self,
        time: float,
        system_state: state.SystemState,
    ):
        O_vector = self.psi_selection.eval_PSIs_on_state(system_state=system_state)

        E_loc = self.get_base_energy(
            system_state=system_state
        ) + self.eval_V_n_expectation(
            time=time, eta_vec=self.eta_vec, system_state=system_state
        )

        return O_vector, E_loc

    def get_H_n(
        self,
        time: float,
        system_state: state.SystemState,
    ) -> np.complex128:
        #! We require the scaling of the samples. Because of that while initializing, this check can not run
        if (
            not self.is_initializing and self.current_time_cache != time
        ):  # float comparison is ok, because float stems from same float normally, so this is bit-accurate
            raise Exception("The Hamiltonian is not initialized for the requested time")

        PSI_vector = self.psi_selection.eval_PSIs_on_state(system_state=system_state)

        # must this include the scaling factor psi_0? -> NO, as we divide at the appropriate places, which is equivalent to doing the ln addition
        # and all other places the psi_0's cancel in the differences, because of uniform initial state assumption
        # E_0 will be inserted because the E_heff call does that
        return np.dot(self.eta_vec, PSI_vector)

    def get_number_of_eta_parameters(self) -> int:
        # each eta has one PSI
        return self.psi_selection.get_number_of_PSIs()

    def get_initialized_eta_vec(self) -> ETAVecType:
        real_part = np.array(
            [
                self.random_generator.normal(sigma=self.init_sigma)
                for _ in range(self.get_number_of_eta_parameters())
            ],
            dtype=np.complex128,
        )
        complex_part = np.array(
            [
                self.random_generator.normal(sigma=self.init_sigma)
                for _ in range(self.get_number_of_eta_parameters())
            ],
            dtype=np.complex128,
        )

        return real_part + 1j * complex_part

    def eval_V_n_expectation(
        self, time: float, eta_vec: ETAVecType, system_state: state.SystemState
    ):
        collecting_sum = np.complex128(0)

        for l in range(
            self.psi_selection.system_geometry.get_number_sites_wo_spin_degree()
        ):
            for m in self.psi_selection.system_geometry.get_nearest_neighbor_indices(l):
                for up in [True, False]:
                    # H^0(n,t) = -i * E_0(n) * t + ln(psi_0(s))
                    # H_VCN is missing ln(psi_0) part: as all psi_0 are equal, they cancel

                    collecting_sum += np.exp(
                        np.dot(
                            eta_vec,
                            # needs minus, because formula and convention here inverted
                            -self.psi_selection.eval_PSI_differences_on_l_to_m_hopped_state(
                                before_swap_system_state=system_state,
                                l=l,
                                m=m,
                                spins_up=up,
                            ),
                        )
                        # needs minus, because formula and convention here inverted, but cancels with the -i
                        + 1j
                        * time
                        * self.get_base_energy_difference_l_to_m_hopping(
                            before_swap_system_state=system_state,
                            l=l,
                            m=m,
                            spins_up=up,
                        )
                    )

        return -self.psi_selection.J * collecting_sum

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return super().get_log_info(
            {
                "type": "VCNHardCoreBosonicHamiltonian",
                "eta_calculation_sampler": self.eta_calculation_sampler.get_log_info(),
                "psi_selection": self.psi_selection.get_log_info(),
                "random_generator": self.random_generator.get_log_info(),
                "init_sigma": self.init_sigma,
                "pseudo_inverse_cutoff": self.pseudo_inverse_cutoff,
                "variational_step_fraction_multiplier": self.variational_step_fraction_multiplier,
                **additional_info,
            }
        )


class VCNHardCoreBosonicHamiltonianAnalyticalParamsFirstOrder(
    VCNHardCoreBosonicHamiltonian
):
    eta_vec: ETAVecType  # typechecker whines around, when I use inline type annotation for this...

    def __init__(
        self,
        U: float,
        E: float,
        J: float,
        phi: float,
        initial_system_state: state.InitialSystemState,
        psi_selection: PSISelection,
        random_generator: RandomGenerator,
        init_sigma: float,
        eta_calculation_sampler: "sampler.GeneralSampler",
        pseudo_inverse_cutoff: float,
        variational_step_fraction_multiplier: int,
    ):
        super().__init__(
            U,
            E,
            J,
            phi,
            initial_system_state,
            psi_selection,
            random_generator,
            init_sigma,
            eta_calculation_sampler,
            pseudo_inverse_cutoff,
            variational_step_fraction_multiplier,
        )

        if not isinstance(psi_selection, ChainDirectionDependentAllSameFirstOrder):
            raise Exception(
                'The Hamiltonian has hard-coded "variational" parameters, it can only be used with the correct comparison PSI-Selection'
            )

    def initialize(
        self,
        time: float,
    ):
        self.is_initializing = True

        # FIRST ORDER ANALYTICAL COEFFICIENTS FOR COMPARISON
        eps_0 = self.E * self.psi_selection.system_geometry.get_eps_multiplier(
            0, self.phi, self.sin_phi, self.cos_phi
        )
        eps_1 = self.E * self.psi_selection.system_geometry.get_eps_multiplier(
            1, self.phi, self.sin_phi, self.cos_phi
        )
        self.eta_vec = np.array(
            [
                np.expm1(1j * (eps_0 - eps_1) * time) / (eps_0 - eps_1),
                np.expm1(1j * (eps_0 - eps_1 + self.U) * time)
                / (eps_0 - eps_1 + self.U),
                np.expm1(1j * (eps_0 - eps_1 - self.U) * time)
                / (eps_0 - eps_1 - self.U),
                np.expm1(1j * (eps_1 - eps_0) * time) / (eps_1 - eps_0),
                np.expm1(1j * (eps_1 - eps_0 + self.U) * time)
                / (eps_1 - eps_0 + self.U),
                np.expm1(1j * (eps_1 - eps_0 - self.U) * time)
                / (eps_1 - eps_0 - self.U),
            ]
        )

        # finished and the result is that trained self.eta_vec
        self.current_time_cache = time
        self.is_initializing = False

    def get_log_info(
        self, additional_info: Dict[str, Union[float, str, Dict[str, Any]]] = {}
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        return super().get_log_info(
            {
                "type": "VCNHardCoreBosonicHamiltonianAnalyticalParamsFirstOrder",  # overwrites this
                **additional_info,
            }
        )
