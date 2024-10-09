from typing import Dict, Union, Any
from abc import ABC, abstractmethod
import state
import systemgeometry
import hamiltonian
import numpy as np
from concurrence import calculate_concurrence


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
    ) -> np.complex128 | np.ndarray:
        """If the post_process_necessary() returns False, this returns numbers directly.
        If it returns True, this will return an array that will be converted to a number by applying post_process()
        """

    def post_process_necessary(self) -> bool:
        return False

    def post_process(self, value: np.ndarray) -> np.complex128:
        return value[0]

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
        self.site_os = system_geometry.get_opposite_spin_index(self.site)

    def get_expectation_value(
        self, time: float, system_state: state.SystemState
    ) -> np.complex128:
        _ = time  # time is not used
        system_state_array = system_state.get_state_array()

        return np.complex128(
            system_state_array[self.site] * system_state_array[self.site_os]
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
                # required that the direction dependent operation is hermitian
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


class Concurrence(Observable):

    def __init__(
        self,
        site_index_from: int,
        spin_up_from: bool,
        site_index_to: int,
        spin_up_to: bool,
        system_hamiltonian: hamiltonian.Hamiltonian,
        system_geometry: systemgeometry.SystemGeometry,
        perform_checks: bool = False,
    ):
        super().__init__()

        if site_index_from < 0 or site_index_to < 0:
            raise Exception("Site must be at least 0")

        domain_size = system_geometry.get_number_sites_wo_spin_degree()
        if site_index_from >= domain_size or site_index_to >= domain_size:
            raise Exception(f"Site must be smaller than {domain_size} to fit")

        # makes sense to check this also further than nearest neighbors

        self.name_from = str(site_index_from) + (" up" if spin_up_from else " down")
        self.name_to = str(site_index_to) + (" up" if spin_up_to else " down")

        self.index_l = site_index_from
        self.up_sigma = spin_up_from
        self.index_m = site_index_to
        self.up_sigma_prime = spin_up_to

        self.use_index_from = site_index_from  # l
        if not spin_up_from:
            self.use_index_from = system_geometry.get_opposite_spin_index(
                site_index_from
            )
        self.use_index_to = site_index_to  # m
        if not spin_up_to:
            self.use_index_to = system_geometry.get_opposite_spin_index(site_index_to)

        self.system_hamiltonian = system_hamiltonian

        self.perform_checks = perform_checks

    def get_expectation_value(
        self, time: float, system_state: state.SystemState
    ) -> np.complex128:
        system_state_array = system_state.get_state_array()

        site_occ_l_sigma_sign = 2 * system_state_array[self.use_index_from] - 1
        site_occ_m_sigma_prime_sign = 2 * system_state_array[self.use_index_to] - 1
        # notice minus for e^H_eff_tilde-H_eff factors: difference is wrong way round from function. need (flipped - original)
        e_to_diff_l_sigma = np.eps(
            -self.system_hamiltonian.get_H_eff_difference_flipping(
                time=time,
                flipping_up=self.up_sigma,
                flipping_index=self.index_l,
                before_swap_system_state=system_state,
            )
        )
        e_to_diff_m_sigma_prime = np.eps(
            -self.system_hamiltonian.get_H_eff_difference_flipping(
                time=time,
                flipping_up=self.up_sigma_prime,
                flipping_index=self.index_m,
                before_swap_system_state=system_state,
            )
        )
        e_to_diff_both = np.eps(
            -self.system_hamiltonian.get_H_eff_difference_double_flipping(
                time=time,
                flipping1_up=self.up_sigma,
                flipping1_index=self.index_l,
                flipping2_up=self.up_sigma_prime,
                flipping2_index=self.index_m,
                before_swap_system_state=system_state,
            )
        )

        return np.array(
            [
                [
                    1,  # 0(l,sigma) 0(m,sigma_p)
                    e_to_diff_m_sigma_prime,  # 0(l,sigma) x(m,sigma_p)
                    e_to_diff_m_sigma_prime
                    * -1j
                    * site_occ_m_sigma_prime_sign,  # 0(l,sigma) y(m,sigma_p)
                    site_occ_m_sigma_prime_sign,  # 0(l,sigma) z(m,sigma_p)
                ],
                [
                    e_to_diff_l_sigma,  # x(l,sigma) 0(m,sigma_p)
                    e_to_diff_both,  # x(l,sigma) x(m,sigma_p)
                    e_to_diff_both
                    * site_occ_m_sigma_prime_sign
                    * 1j,  # x(l,sigma) y(m,sigma_p)
                    e_to_diff_l_sigma
                    * site_occ_m_sigma_prime_sign,  # x(l,sigma) z(m,sigma_p)
                ],
                [
                    e_to_diff_l_sigma
                    * -1j
                    * site_occ_l_sigma_sign,  # y(l,sigma) 0(m,sigma_p)
                    e_to_diff_both
                    * site_occ_l_sigma_sign
                    * 1j,  # y(l,sigma) x(m,sigma_p)
                    e_to_diff_both
                    * (
                        -1 * site_occ_l_sigma_sign * site_occ_m_sigma_prime_sign
                    ),  # y(l,sigma) y(m,sigma_p)
                    e_to_diff_l_sigma
                    * -1j
                    * site_occ_l_sigma_sign
                    * site_occ_m_sigma_prime_sign,  # y(l,sigma) z(m,sigma_p)
                ],
                [
                    site_occ_l_sigma_sign,  # z(l,sigma) 0(m,sigma_p)
                    site_occ_l_sigma_sign
                    * e_to_diff_m_sigma_prime,  # z(l,sigma) x(m,sigma_p)
                    site_occ_l_sigma_sign
                    * e_to_diff_m_sigma_prime
                    * -1j
                    * site_occ_m_sigma_prime_sign,  # z(l,sigma) y(m,sigma_p)
                    site_occ_l_sigma_sign
                    * site_occ_m_sigma_prime_sign,  # z(l,sigma) z(m,sigma_p)
                ],
            ],
            dtype=np.complex128,
        )

    def post_process_necessary(self) -> bool:
        # this must store the multiplied and non multiplied occupations seperately, because the averaging of products is not equal the products of averages
        return True

    def post_process(self, value: np.ndarray) -> np.complex128:
        return calculate_concurrence(value, do_checks=self.perform_checks)

    def get_label(self) -> str:
        return f"Concurrence between {self.name_from} and {self.name_to}"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {
            "type": "SpinCurrent",
            "label": self.get_label(),
            "use_index_from": self.use_index_from,
            "use_index_to": self.use_index_to,
            "perform_checks": self.perform_checks,
        }
