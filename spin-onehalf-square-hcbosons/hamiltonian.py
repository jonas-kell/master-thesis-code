from abc import ABC, abstractmethod
import state
import numpy as np
from typing import Dict, List, Tuple
from enum import Enum


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
        self.cos_phi = np.cos(self.phi)
        self.sin_phi = np.sin(self.phi)

    @abstractmethod
    def get_base_energy(
        self, system_state_object: state.SystemState, system_state_array: np.ndarray
    ) -> float:
        """
        system_state_object is ONLY for index/eps calculations
        all state data will be taken from the system_state_array
        """
        pass

    @abstractmethod
    def get_H_n(
        self,
        time: float,
        system_state_object: state.SystemState,
        system_state_array: np.ndarray,
    ) -> float:
        """
        system_state_object is ONLY for index/eps calculations
        all state data will be taken from the system_state_array
        """
        pass


class VPartsMapping(Enum):
    ClCHm = "a"  # red
    DlDHm = "b"  # red
    ClCmCHlCHmDlDHm = "c"  # green
    ClCHmDlDmDHlDHm = "d"  # green
    ClCHlDlDHm = "e"  # violet
    ClCHmDlDHl = "f"  # violet
    CmCHmDlDHm = "g"  # yellow
    ClCHmDmDHm = "g"  # yellow


class HardcoreBosonicHamiltonian(Hamiltonian):
    def __init__(
        self,
        U: float,
        E: float,
        J: float,
        phi: float,
    ):
        super().__init__(U=U, E=E, J=J, phi=phi)

    def get_base_energy(
        self, system_state_object: state.SystemState, system_state_array: np.ndarray
    ) -> float:
        u_count = 0.0
        eps_collector = 0.0

        for index in range(system_state_object.get_number_sites_wo_spin_degree()):
            # opposite spin
            index_os = system_state_object.get_opposite_spin_index(index)

            # count number of double occupancies
            u_count += system_state_array[index] * system_state_array[index_os]

            # collect epsilon values
            eps_collector += system_state_array[
                index
            ] * system_state_object.get_eps_multiplier(
                index=index, phi=self.phi, sin_phi=self.sin_phi, cos_phi=self.cos_phi
            ) + system_state_array[
                index_os
            ] * system_state_object.get_eps_multiplier(
                index=index_os, phi=self.phi, sin_phi=self.sin_phi, cos_phi=self.cos_phi
            )

        return self.U * u_count + self.E * eps_collector

    def get_H_n(
        self,
        time: float,
        system_state_object: state.SystemState,
        system_state_array: np.ndarray,
    ) -> float:
        # TODO generate t K states that could potentially overlap when summing over the <l,m>
        pass

    def V_parts(
        self, system_state_object: state.SystemState, system_state_array: np.ndarray
    ) -> Dict[VPartsMapping, List[Tuple[int, int, np.ndarray]]]:
        """
        returns Tuple[l,m,K] The neighbor summation indices l&m and the resulting state K where a match was made
        """
        number_sites = system_state_object.get_number_sites_wo_spin_degree()
        result: Dict[VPartsMapping, List[Tuple[int, int, np.ndarray]]] = {}
        for val in VPartsMapping:
            result[val] = []

        for l in range(number_sites):
            l_os = system_state_object.get_opposite_spin_index(l)

            index_neighbors = system_state_object.get_nearest_neighbor_indices(l)
            index_os_neighbors = system_state_object.get_nearest_neighbor_indices(l_os)

            for m, m_os in zip(index_neighbors, index_os_neighbors):
                # The operators act left onto <system_state_array|operator|output K>

                m_to_l_hopped_state_array = None

                def get_m_to_l_hopped_state_array() -> np.ndarray:
                    nonlocal m_to_l_hopped_state_array
                    if m_to_l_hopped_state_array is None:
                        m_to_l_hopped_state_array = system_state_array.copy()
                        m_to_l_hopped_state_array[l] = 1
                        m_to_l_hopped_state_array[m] = 0
                    return m_to_l_hopped_state_array

                os_m_to_l_hopped_state_array = None

                def get_os_m_to_l_hopped_state_array() -> np.ndarray:
                    nonlocal os_m_to_l_hopped_state_array
                    if os_m_to_l_hopped_state_array is None:
                        os_m_to_l_hopped_state_array = system_state_array.copy()
                        os_m_to_l_hopped_state_array[l_os] = 1
                        os_m_to_l_hopped_state_array[m_os] = 0
                    return os_m_to_l_hopped_state_array

                # ClCHm: c_l * c#_m
                if system_state_array[l] == 0 and system_state_array[m] == 1:
                    result[VPartsMapping.ClCHm].append(
                        (l, m, get_m_to_l_hopped_state_array())
                    )

                # DlDHm: d_l * d#_m
                if system_state_array[l_os] == 0 and system_state_array[l_os] == 1:
                    result[VPartsMapping.DlDHm].append(
                        (l, m, get_os_m_to_l_hopped_state_array())
                    )

                # ClCmCHlCHmDlDHm: c_l * c_m * c#_l * c#_m * d_l * d#_m
                if (
                    system_state_array[l] == 0
                    and system_state_array[m] == 0
                    and system_state_array[l_os] == 0
                    and system_state_array[m_os] == 1
                ):
                    result[VPartsMapping.ClCmCHlCHmDlDHm].append(
                        (l, m, get_os_m_to_l_hopped_state_array())
                    )

                # ClCHmDlDmDHlDHm: c_l * c#_m * d_l * d_m * d#_l * d#_m
                if (
                    system_state_array[l_os] == 0
                    and system_state_array[m_os] == 0
                    and system_state_array[l] == 0
                    and system_state_array[m] == 1
                ):
                    result[VPartsMapping.ClCHmDlDmDHlDHm].append(
                        (l, m, get_m_to_l_hopped_state_array())
                    )

                # ClCHlDlDHm: c_l * c#_l * d_l * d#_m
                if (
                    system_state_array[l] == 0
                    and system_state_array[l_os] == 0
                    and system_state_array[m_os] == 1
                ):
                    result[VPartsMapping.ClCHlDlDHm].append(
                        (l, m, get_os_m_to_l_hopped_state_array())
                    )

                # ClCHmDlDHl: c_l * c#_m * d_l * d#_l
                if (
                    system_state_array[l_os] == 0
                    and system_state_array[l] == 0
                    and system_state_array[m] == 1
                ):
                    result[VPartsMapping.ClCHmDlDHl].append(
                        (l, m, get_m_to_l_hopped_state_array())
                    )

                # CmCHmDlDHm: c_m * c#_m * d_l * d#_m
                if (
                    system_state_array[m] == 0
                    and system_state_array[l_os] == 0
                    and system_state_array[m_os] == 1
                ):
                    result[VPartsMapping.CmCHmDlDHm].append(
                        (l, m, get_os_m_to_l_hopped_state_array())
                    )

                # ClCHmDmDHm: c_l * c#_m * d_m * d#_m
                if (
                    system_state_array[m_os] == 0
                    and system_state_array[l] == 0
                    and system_state_array[m] == 1
                ):
                    result[VPartsMapping.ClCHmDmDHm].append(
                        (l, m, get_m_to_l_hopped_state_array())
                    )

        return result
