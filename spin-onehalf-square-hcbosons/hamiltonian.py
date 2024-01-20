from abc import ABC, abstractmethod
import state
import numpy as np


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
    def get_base_energy(self) -> float:
        pass


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
        """
        system_state_object is ONLY for index/eps calculations
        all state data will be taken from the system_state_array
        """
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


# for neighbor in [*index_neighbors, *index_os_neighbors]:
