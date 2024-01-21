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
    ClCHm = "a"


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

                # c_l*c#_m
                if system_state_array[l] == 0 and system_state_array[m] == 1:
                    tmp = system_state_array.copy()
                    tmp[l] = 1
                    tmp[m] = 0
                    result[VPartsMapping.ClCHm].append((l, m, tmp))
        return result
