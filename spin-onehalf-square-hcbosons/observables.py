from abc import ABC, abstractmethod
import state
import systemgeometry
from typing import Dict, Union, Any
import hamiltonian
import numpy as np


class Observable(ABC):
    def __init__(
        self,
    ):
        pass

    # This should only require returning float. As observables only return real values.
    # but we give back the imaginary part, to check that it really cancels
    @abstractmethod
    def get_expectation_value(
        self, time: float, system_state: state.SystemState
    ) -> np.complex128:
        pass

    @abstractmethod
    def get_label(self) -> str:
        pass

    @abstractmethod
    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        pass


class DoubleOccupationFraction(Observable):
    def __init__(
        self,
    ):
        super().__init__()

    def get_expectation_value(
        self, time: float, system_state: state.SystemState
    ) -> np.complex128:
        _ = time  # time is not used
        nr_sites = system_state.get_number_sites_wo_spin_degree()
        system_state_array = system_state.get_state_array()
        domain_size = system_state.get_number_sites_wo_spin_degree()

        running_sum = 0
        for i in range(nr_sites):
            i_os = system_state.get_opposite_spin_index(i)

            # only because occupation is either 1 or 0
            running_sum += system_state_array[i] * system_state_array[i_os]

        return np.complex128(running_sum / domain_size)

    def get_label(self) -> str:
        return "Average amount of double Occupation"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {"type": "DoubleOccupationFraction", "label": self.get_label()}


class DoubleOccupationAtSite(Observable):

    def __init__(self, site: int, system_geometry: systemgeometry.SystemGeometry):
        super().__init__()

        if site < 0:
            raise Exception("Site must be at least 0")

        domain_size = system_geometry.get_number_sites_wo_spin_degree()
        if site >= domain_size:
            raise Exception(f"Site must be smaller than {domain_size} to fit")

        self.site = site

    def get_expectation_value(
        self, time: float, system_state: state.SystemState
    ) -> np.complex128:
        _ = time  # time is not used
        system_state_array = system_state.get_state_array()

        return np.complex128(
            system_state_array[self.site]
            * system_state_array[system_state.get_opposite_spin_index(self.site)]
        )

    def get_label(self) -> str:
        return f"Double Occupation at site {self.site}"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {
            "type": "DoubleOccupationAtSite",
            "label": self.get_label(),
            "site": self.site,
        }


class SpinCurrent(Observable):

    def __init__(
        self,
        site_index_from: int,
        site_index_to: int,
        spin_up: bool,
        system_hamiltonian: hamiltonian.Hamiltonian,
        system_geometry: systemgeometry.SystemGeometry,
        direction_dependent: bool = True,
    ):
        super().__init__()

        if site_index_from < 0 or site_index_to < 0:
            raise Exception("Site must be at least 0")

        domain_size = system_geometry.get_number_sites_wo_spin_degree()
        if site_index_from >= domain_size or site_index_to >= domain_size:
            raise Exception(f"Site must be smaller than {domain_size} to fit")

        if not site_index_to in system_geometry.get_nearest_neighbor_indices(
            site_index_from
        ):
            raise Exception(f"site_index_from must be a neighbor of site_index_to")

        self.site_index_from = site_index_from  # l
        self.site_index_to = site_index_to  # m
        self.spin_up = spin_up
        self.system_hamiltonian = system_hamiltonian
        self.direction_dependent = direction_dependent
        self.mod_safe = domain_size

    def get_expectation_value(
        self, time: float, system_state: state.SystemState
    ) -> np.complex128:
        system_state_array = system_state.get_state_array()

        site_occ_l = system_state_array[self.site_index_from]
        site_occ_m = system_state_array[self.site_index_to]
        forward_swap_condition = site_occ_l == 1 and site_occ_m == 0
        backward_swap_condition = site_occ_l == 0 and site_occ_m == 1

        res: np.complex128 = np.complex128(0)
        if forward_swap_condition or backward_swap_condition:
            H_eff_difference, psi_factor = (
                self.system_hamiltonian.get_H_eff_difference_swapping(
                    time=time,
                    before_swap_system_state=system_state,
                    sw1_index=(self.site_index_from % self.mod_safe),
                    sw2_index=(self.site_index_to % self.mod_safe),
                    sw1_up=self.spin_up,
                    sw2_up=self.spin_up,
                )
            )

            if forward_swap_condition:
                res += np.exp(H_eff_difference) * psi_factor

            if backward_swap_condition:
                if self.direction_dependent:
                    # other direction = other sign
                    res -= np.exp(H_eff_difference) * psi_factor
                else:
                    # both directions contribute with same sign
                    res += np.exp(H_eff_difference) * psi_factor

            if self.direction_dependent:
                # required that the direction dependent operation is hermetian
                res *= 1j

        # Upstream functions check that the imaginary part of this cancels
        return -self.system_hamiltonian.J * res

    def get_label(self) -> str:
        return f"Spin Current {'(signed)' if self.direction_dependent else ''} from {self.site_index_from} to {self.site_index_to} ({'up' if self.spin_up else 'down'})"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {
            "type": "SpinCurrent",
            "label": self.get_label(),
            "site_index_from": self.site_index_from,
            "site_index_to": self.site_index_to,
            "spin_up": self.spin_up,
            "direction_dependent": self.direction_dependent,
        }
