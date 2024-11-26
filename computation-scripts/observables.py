from typing import Dict, Union, Any
from abc import ABC, abstractmethod
import state
import systemgeometry
import hamiltonian
import numpy as np
from concurrence import (
    concurrence_of_density_matrix,
    concurrence_of_density_matrix_assym,
    get_reduced_density_matrix_in_z_basis_from_observations,
)


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


# this separation is not type-safely/properly extenable/generalizable.
class MeasurableObservable(Observable):
    pass


# this is a hacky way of doing it, but I cannot be bothered to adapt the other observable-driven plotting/calculation systems, sorry not sorry
class HamiltonianProperty(Observable):
    def __init__(self, ham: hamiltonian.Hamiltonian):
        self.hamiltonian = ham


class Energy(MeasurableObservable):
    def __init__(
        self, ham: hamiltonian.Hamiltonian, geometry: systemgeometry.SystemGeometry
    ):
        self.hamiltonian = ham
        self.number_of_sites = geometry.get_number_sites_wo_spin_degree()

        super().__init__()

    def get_expectation_value(
        self, time: float, system_state: state.SystemState
    ) -> np.complex128:
        E_0 = self.hamiltonian.get_base_energy(system_state=system_state)

        v_aggregator = np.complex128(0)
        for l in range(self.number_of_sites):
            for m in system_state.get_nearest_neighbor_indices(l):
                for spin_up in [True, False]:
                    if spin_up:
                        use_l = l
                        use_m = m
                    else:
                        use_l = system_state.get_opposite_spin_index(l)
                        use_m = system_state.get_opposite_spin_index(m)

                    if (
                        system_state.getget_state_array()[use_l]
                        != system_state.get_state_array()[use_m]
                    ):
                        # add the psi difference
                        e_diff, psi_fact = (
                            self.hamiltonian.get_H_eff_difference_double_flipping(
                                time=time,
                                flipping1_up=spin_up,
                                flipping1_index=l,
                                flipping2_up=spin_up,
                                flipping2_index=m,
                                before_swap_system_state=system_state,
                            )
                        )
                        v_aggregator += np.exp(-e_diff) / psi_fact

        return (E_0 - self.hamiltonian.J * v_aggregator) / self.number_of_sites

    def get_label(self) -> str:
        return "Energy per site"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {"type": "Energy", "label": self.get_label()}


class EnergyVariance(MeasurableObservable):
    def __init__(
        self, ham: hamiltonian.Hamiltonian, geometry: systemgeometry.SystemGeometry
    ):
        self.hamiltonian = ham
        self.number_of_sites = geometry.get_number_sites_wo_spin_degree()
        self.geometry = geometry
        geometry.init_index_overlap_circle_cache(2)
        # 2 is known to be a good value for all apptoximations we do here
        # for approximations to higher degrees, it might be necessary to change this
        # we cannot save computation, by selecting 1 here for first order perturbations (unclear, what goes of of that range)
        # see test setup in ./compareenergyvariance.py

        super().__init__()

    def eval_variance_op(
        self,
        time: float,
        system_state: state.SystemState,
        l: int,
        m: int,
        a: int,
        b: int,
        sigma_up: bool,
        mu_up: bool,
    ) -> np.complex128:
        # extract the relevant occupations from the array
        if sigma_up:
            use_l_index = l
            use_m_index = m
        else:
            use_l_index = system_state.get_opposite_spin_index(l)
            use_m_index = system_state.get_opposite_spin_index(m)
        if mu_up:
            use_a_index = a
            use_b_index = b
        else:
            use_a_index = system_state.get_opposite_spin_index(a)
            use_b_index = system_state.get_opposite_spin_index(b)

        initial_occ_l = system_state.get_state_array()[use_l_index]
        initial_occ_m = system_state.get_state_array()[use_m_index]
        initial_occ_a = system_state.get_state_array()[use_a_index]
        initial_occ_b = system_state.get_state_array()[use_b_index]

        # positive part: 4-way flips
        final_occ_l_square = initial_occ_l
        final_occ_m_square = initial_occ_m
        final_occ_a_square = initial_occ_a
        final_occ_b_square = initial_occ_b
        # lm might influence the ab occupations, because in hte f-way op it is left-er and they act left
        if sigma_up == mu_up:
            if l == a or m == a:
                final_occ_a_square = 1 - final_occ_a_square
            if l == b or m == b:
                final_occ_b_square = 1 - final_occ_b_square

        # negative part: 2 x 2-way flip
        final_occ_l_double = initial_occ_l
        final_occ_m_double = initial_occ_m
        final_occ_a_double = initial_occ_a
        final_occ_b_double = initial_occ_b
        # both are separate, l<->m, a<->b, so no modifications necessary

        # evaluate the exp-differences here (do not do expensive exp calculation, if control term is 0)
        factor_square = (
            final_occ_l_square
            * (1 - final_occ_m_square)
            * final_occ_a_square
            * (1 - final_occ_b_square)
        )
        exp_JJ = 0
        if factor_square:
            # "get_exp_H_eff_difference_quadruple_flipping":
            # (inlined to avoid re-computation of use_indices)

            # it can never be (external and logical assertion) use_l_index == use_a_index or use_m_index == use_b_index

            if use_l_index == use_b_index and use_m_index == use_a_index:
                # nothing happens, because the operators revert their changes cross-wise
                exp_JJ = 1.0  # exp^0
            elif use_l_index == use_b_index:  # use_m_index != use_a_index
                # l and b effect cancels -> reduced to a two-way flipping computation
                exp_JJ = np.exp(
                    -self.hamiltonian.get_H_eff_difference_double_flipping(
                        time=time,
                        flipping1_up=sigma_up,
                        flipping1_index=m,
                        flipping2_up=mu_up,
                        flipping2_index=a,
                        before_swap_system_state=system_state,
                    )[0]
                )
            elif use_m_index == use_a_index:  # use_l_index != use_b_index
                # m and a effect cancels -> reduced to a two-way flipping computation
                exp_JJ = np.exp(
                    -self.hamiltonian.get_H_eff_difference_double_flipping(
                        time=time,
                        flipping1_up=sigma_up,
                        flipping1_index=l,
                        flipping2_up=mu_up,
                        flipping2_index=b,
                        before_swap_system_state=system_state,
                    )[0]
                )
            else:
                # all use_indices use_l, use_m, use_a, use_b are different -> the true 4-way flip

                # ! this substitutes two two way flip differences, to avoid needing to implement 4-way flip logic
                modify_arr = system_state.get_state_array()

                # N(4-flip) - N(no flip) = [N(4-flip) - N(2-flip)] + [N(no flip) - N(2-flip)]

                # flip in-place for computation
                modify_arr[use_l_index] = 1 - modify_arr[use_l_index]
                modify_arr[use_m_index] = 1 - modify_arr[use_m_index]
                a_result = self.hamiltonian.get_H_eff_difference_double_flipping(
                    time=time,
                    flipping1_up=mu_up,
                    flipping1_index=a,
                    flipping2_up=mu_up,
                    flipping2_index=b,
                    before_swap_system_state=system_state,  # this was changed by reference
                )[0]
                # flip back, function doesn't modify state permanently, change is only in this section
                modify_arr[use_l_index] = 1 - modify_arr[use_l_index]
                modify_arr[use_m_index] = 1 - modify_arr[use_m_index]
                b_result = self.hamiltonian.get_H_eff_difference_double_flipping(
                    time=time,
                    flipping1_up=sigma_up,
                    flipping1_index=l,
                    flipping2_up=sigma_up,
                    flipping2_index=m,
                    before_swap_system_state=system_state,
                )[0]

                exp_JJ = np.exp(-a_result - b_result)

            # END of "get_exp_H_eff_difference_quadruple_flipping"
        factor_double = (
            final_occ_l_double
            * (1 - final_occ_m_double)
            * final_occ_a_double
            * (1 - final_occ_b_double)
        )
        exp_lm = 0
        exp_ab = 0
        if factor_double:
            exp_lm = np.exp(
                -self.hamiltonian.get_H_eff_difference_double_flipping(
                    time=time,
                    flipping1_up=sigma_up,
                    flipping1_index=l,
                    flipping2_up=sigma_up,
                    flipping2_index=m,
                    before_swap_system_state=system_state,
                )[0]
            )
            exp_ab = np.exp(
                -self.hamiltonian.get_H_eff_difference_double_flipping(
                    time=time,
                    flipping1_up=mu_up,
                    flipping1_index=a,
                    flipping2_up=mu_up,
                    flipping2_index=b,
                    before_swap_system_state=system_state,
                )[0]
            )

        return (exp_JJ) - (exp_lm * exp_ab)

    def get_expectation_value(
        self, time: float, system_state: state.SystemState
    ) -> np.complex128:
        variance_sum_aggregator = np.complex128(0)
        for sigma_up in [True, False]:
            for mu_up in [True, False]:
                for l, m, a, b in self.geometry.get_index_overlap_circle_tuples():
                    variance_sum_aggregator += self.eval_variance_op(
                        system_state=system_state,
                        l=l,
                        m=m,
                        a=a,
                        b=b,
                        sigma_up=sigma_up,
                        mu_up=mu_up,
                        time=time,
                    )

        return (
            self.hamiltonian.J * self.hamiltonian.J * variance_sum_aggregator
        ) / self.number_of_sites

    def get_label(self) -> str:
        return "Energy Variance per site"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {"type": "EnergyVariance", "label": self.get_label()}


class DoubleOccupationFraction(MeasurableObservable):
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


class DoubleOccupationAtSite(MeasurableObservable):

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


class OccupationAtSite(MeasurableObservable):

    def __init__(
        self, site: int, up: bool, system_geometry: systemgeometry.SystemGeometry
    ):
        super().__init__()

        if site < 0:
            raise Exception("Site must be at least 0")

        domain_size = system_geometry.get_number_sites_wo_spin_degree()
        if site >= domain_size:
            raise Exception(f"Site must be smaller than {domain_size} to fit")

        self.site = site
        self.up = up

        self.site_to_use = self.site
        if not up:
            self.site_to_use = system_geometry.get_opposite_spin_index(self.site)

    def get_expectation_value(
        self, time: float, system_state: state.SystemState
    ) -> np.complex128:
        _ = time  # time is not used
        system_state_array = system_state.get_state_array()

        return np.complex128(system_state_array[self.site_to_use])

    def get_label(self) -> str:
        return f"Occupation at site {self.site}, {'up' if self.up else 'down'}"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {
            "type": "OccupationAtSite",
            "label": self.get_label(),
            "site": self.site,
            "up": self.up,
        }


class SpinCurrent(MeasurableObservable):

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

        self.site_index_from_save = site_index_from % domain_size
        self.site_index_to_save = site_index_to % domain_size
        self.occ_site_index_from = self.site_index_from_save
        self.occ_site_index_to = self.site_index_to_save
        if not self.spin_up:
            self.occ_site_index_from = system_geometry.get_opposite_spin_index(
                self.occ_site_index_from
            )
            self.occ_site_index_to = system_geometry.get_opposite_spin_index(
                self.occ_site_index_to
            )

        self.system_hamiltonian = system_hamiltonian
        self.direction_dependent = direction_dependent

    def get_expectation_value(
        self, time: float, system_state: state.SystemState
    ) -> np.complex128:
        system_state_array = system_state.get_state_array()

        site_occ_l = system_state_array[self.occ_site_index_from]
        site_occ_m = system_state_array[self.occ_site_index_to]
        forward_swap_condition = site_occ_l == 1 and site_occ_m == 0
        disjunct_condition = site_occ_l != site_occ_m

        res: np.complex128 = np.complex128(0)
        if disjunct_condition:
            H_eff_difference, psi_factor = (
                self.system_hamiltonian.get_H_eff_difference_swapping(
                    time=time,
                    before_swap_system_state=system_state,
                    sw1_index=self.site_index_from_save,
                    sw2_index=self.site_index_to_save,
                    sw1_up=self.spin_up,
                    sw2_up=self.spin_up,
                )
            )

            # notice minus for e^H_eff_tilde-H_eff factors: difference is wrong way round from function. need (swapped - original)
            # therefore also (x/psi) not (x*psi)

            if forward_swap_condition:
                res += np.exp(-H_eff_difference) / psi_factor
            else:
                if self.direction_dependent:
                    # other direction = other sign
                    res -= np.exp(-H_eff_difference) / psi_factor
                else:
                    # both directions contribute with same sign
                    res += np.exp(-H_eff_difference) / psi_factor

            # this can be indented to here, as if it was one more layer further out there would be only value = 0
            if self.direction_dependent:
                # required to make the direction dependent operation hermitian
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


class SpinCurrentFlipping(SpinCurrent):

    def get_expectation_value(
        self, time: float, system_state: state.SystemState
    ) -> np.complex128:
        system_state_array = system_state.get_state_array()

        site_occ_l = system_state_array[self.occ_site_index_from]
        site_occ_m = system_state_array[self.occ_site_index_to]
        forward_swap_condition = site_occ_l == 1 and site_occ_m == 0
        disjunct_condition = site_occ_l != site_occ_m

        res: np.complex128 = np.complex128(0)
        if disjunct_condition:
            H_eff_difference, psi_factor = (
                self.system_hamiltonian.get_H_eff_difference_double_flipping(
                    time=time,
                    before_swap_system_state=system_state,
                    flipping1_index=self.site_index_from_save,
                    flipping2_index=self.site_index_to_save,
                    flipping1_up=self.spin_up,
                    flipping2_up=self.spin_up,
                )
            )

            # notice minus for e^H_eff_tilde-H_eff factors: difference is wrong way round from function. need (swapped - original)
            # therefore also (x/psi) not (x*psi)

            if forward_swap_condition:
                res += np.exp(-H_eff_difference) / psi_factor
            else:
                if self.direction_dependent:
                    # other direction = other sign
                    res -= np.exp(-H_eff_difference) / psi_factor
                else:
                    # both directions contribute with same sign
                    res += np.exp(-H_eff_difference) / psi_factor

            # this can be indented to here, as if it was one more layer further out there would be only value = 0
            if self.direction_dependent:
                # required to make the direction dependent operation hermitian
                res *= 1j

        # Upstream functions check that the imaginary part of this cancels
        return -self.system_hamiltonian.J * res

    def get_label(self) -> str:
        return f"Spin Current Flipping {'(signed)' if self.direction_dependent else ''} from {self.site_index_from} to {self.site_index_to} ({'up' if self.spin_up else 'down'})"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {
            "type": "SpinCurrentFlipping",
            "label": self.get_label(),
            "site_index_from": self.site_index_from,
            "site_index_to": self.site_index_to,
            "spin_up": self.spin_up,
            "direction_dependent": self.direction_dependent,
        }


class ReducedDensityMatrixMeasurement(MeasurableObservable):

    def __init__(
        self,
        site_index_from: int,
        spin_up_from: bool,
        site_index_to: int,
        spin_up_to: bool,
        system_hamiltonian: hamiltonian.Hamiltonian,
        system_geometry: systemgeometry.SystemGeometry,
        perform_checks: bool = False,
        check_threshold: float = 1e-4,
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
        self.check_threshold = check_threshold

    def get_expectation_value(
        self, time: float, system_state: state.SystemState
    ) -> np.complex128:
        system_state_array = system_state.get_state_array()

        site_occ_l_sigma_sign = 2 * system_state_array[self.use_index_from] - 1
        site_occ_m_sigma_prime_sign = 2 * system_state_array[self.use_index_to] - 1
        # notice minus for e^H_eff_tilde-H_eff factors: difference is wrong way round from function. need (flipped - original)
        # therefore also /psi not * psi
        l_sigma_diff, l_sigma_psi = (
            self.system_hamiltonian.get_H_eff_difference_flipping(
                time=time,
                flipping_up=self.up_sigma,
                flipping_index=self.index_l,
                before_swap_system_state=system_state,
            )
        )
        e_to_diff_l_sigma = np.exp(-l_sigma_diff) / l_sigma_psi
        m_sigma_prime_diff, m_sigma_prime_psi = (
            self.system_hamiltonian.get_H_eff_difference_flipping(
                time=time,
                flipping_up=self.up_sigma_prime,
                flipping_index=self.index_m,
                before_swap_system_state=system_state,
            )
        )
        e_to_diff_m_sigma_prime = np.exp(-m_sigma_prime_diff) / m_sigma_prime_psi
        both_diff, both_psi = (
            self.system_hamiltonian.get_H_eff_difference_double_flipping(
                time=time,
                flipping1_up=self.up_sigma,
                flipping1_index=self.index_l,
                flipping2_up=self.up_sigma_prime,
                flipping2_index=self.index_m,
                before_swap_system_state=system_state,
            )
        )
        e_to_diff_both = np.exp(-both_diff) / both_psi

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
                    * -1j,  # x(l,sigma) y(m,sigma_p)
                    e_to_diff_l_sigma
                    * site_occ_m_sigma_prime_sign,  # x(l,sigma) z(m,sigma_p)
                ],
                [
                    e_to_diff_l_sigma
                    * -1j
                    * site_occ_l_sigma_sign,  # y(l,sigma) 0(m,sigma_p)
                    e_to_diff_both
                    * site_occ_l_sigma_sign
                    * -1j,  # y(l,sigma) x(m,sigma_p)
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
        print(value)  # sigmas
        z_basis_density_matrix_from_measurements = (
            get_reduced_density_matrix_in_z_basis_from_observations(
                value, do_checks=self.perform_checks, threshold=self.check_threshold
            )
        )
        print(z_basis_density_matrix_from_measurements)
        raise Exception(
            "This class should be extended to have more useful post-processing done on the density matrix"
        )

    def get_label(self) -> str:
        return f"Some Matrix Measurement between {self.name_from} and {self.name_to}"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {
            "type": "ReducedDensityMatrixMeasurement",
            "label": self.get_label(),
            "perform_checks": self.perform_checks,
            "index_l": self.index_l,
            "up_sigma": self.up_sigma,
            "index_m": self.index_m,
            "up_sigma_prime": self.up_sigma_prime,
        }


class Concurrence(ReducedDensityMatrixMeasurement):
    def post_process(self, value: np.ndarray) -> np.complex128:
        z_basis_density_matrix_from_measurements = (
            get_reduced_density_matrix_in_z_basis_from_observations(
                value, do_checks=self.perform_checks, threshold=self.check_threshold
            )
        )

        return concurrence_of_density_matrix(z_basis_density_matrix_from_measurements)

    def get_label(self) -> str:
        return f"Concurrence between {self.name_from} and {self.name_to}"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {
            "type": "Concurrence",
            "label": self.get_label(),
            "perform_checks": self.perform_checks,
            "index_l": self.index_l,
            "up_sigma": self.up_sigma,
            "index_m": self.index_m,
            "up_sigma_prime": self.up_sigma_prime,
        }


class ConcurrenceAsymm(ReducedDensityMatrixMeasurement):
    def post_process(self, value: np.ndarray) -> np.complex128:
        z_basis_density_matrix_from_measurements = (
            get_reduced_density_matrix_in_z_basis_from_observations(
                value, do_checks=self.perform_checks, threshold=self.check_threshold
            )
        )

        return concurrence_of_density_matrix_assym(
            z_basis_density_matrix_from_measurements
        )

    def get_label(self) -> str:
        return f"Assym-Concurrence between {self.name_from} and {self.name_to}"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {
            "type": "ConcurrenceAsymm",
            "label": self.get_label(),
            "perform_checks": self.perform_checks,
            "index_l": self.index_l,
            "up_sigma": self.up_sigma,
            "index_m": self.index_m,
            "up_sigma_prime": self.up_sigma_prime,
        }


class Purity(ReducedDensityMatrixMeasurement):
    def post_process(self, value: np.ndarray) -> np.complex128:
        z_basis_density_matrix_from_measurements = (
            get_reduced_density_matrix_in_z_basis_from_observations(
                value, do_checks=self.perform_checks, threshold=self.check_threshold
            )
        )

        return np.trace(
            z_basis_density_matrix_from_measurements
            @ z_basis_density_matrix_from_measurements
        )

    def get_label(self) -> str:
        return f"Purity of red. density matrix of {self.name_from} and {self.name_to}"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {
            "type": "Purity",
            "label": self.get_label(),
            "perform_checks": self.perform_checks,
            "index_l": self.index_l,
            "up_sigma": self.up_sigma,
            "index_m": self.index_m,
            "up_sigma_prime": self.up_sigma_prime,
        }


class PauliMeasurement(ReducedDensityMatrixMeasurement):
    def __init__(
        self,
        site_index_from: int,
        spin_up_from: bool,
        site_index_to: int,
        spin_up_to: bool,
        system_hamiltonian: hamiltonian.Hamiltonian,
        system_geometry: systemgeometry.SystemGeometry,
        perform_checks: bool = False,
        check_threshold: float = 1e-4,
        index_of_pauli_op: int = 0,
    ):
        super().__init__(
            site_index_from=site_index_from,
            spin_up_from=spin_up_from,
            site_index_to=site_index_to,
            spin_up_to=spin_up_to,
            system_hamiltonian=system_hamiltonian,
            system_geometry=system_geometry,
            perform_checks=perform_checks,
            check_threshold=check_threshold,
        )

        if index_of_pauli_op < 0 or index_of_pauli_op > 15:
            raise Exception("Index of op must be between 0 and 15 inclusive")

        self.index_of_pauli_op = index_of_pauli_op

    def post_process(self, value: np.ndarray) -> np.complex128:
        pauli_measurement = value[
            self.index_of_pauli_op // 4, self.index_of_pauli_op % 4
        ]

        # enough other implementations check the real/imag parts of this
        return np.real(pauli_measurement)

    def get_label(self) -> str:
        names = [
            "00",
            "0x",
            "0y",
            "0z",
            "x0",
            "xx",
            "xy",
            "xz",
            "y0",
            "yx",
            "yy",
            "yz",
            "z0",
            "zx",
            "zy",
            "zz",
        ]
        return f"Pauli Measurement {names[self.index_of_pauli_op]} of sites {self.name_from} and {self.name_to}"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {
            "type": "PauliMeasurement",
            "label": self.get_label(),
            "perform_checks": self.perform_checks,
            "index_l": self.index_l,
            "up_sigma": self.up_sigma,
            "index_m": self.index_m,
            "up_sigma_prime": self.up_sigma_prime,
            "index_of_pauli_op": self.index_of_pauli_op,
        }


class VCNFactor(HamiltonianProperty):
    def __init__(
        self, ham: hamiltonian.Hamiltonian, param_index: int, param_real_part: bool
    ):
        super().__init__(ham)

        self.param_index = param_index
        self.param_real_part = param_real_part

        self.can_return_value: bool = False
        self.vcn_hamiltonian: hamiltonian.VCNHardCoreBosonicHamiltonian = None

        if isinstance(self.hamiltonian, hamiltonian.VCNHardCoreBosonicHamiltonian):
            self.can_return_value = True
            self.vcn_hamiltonian = self.hamiltonian
            self.number_eta_params = self.vcn_hamiltonian.get_number_of_eta_parameters()

            if self.param_index < 0 or self.param_index > self.number_eta_params:
                raise Exception(
                    f"VCN Hamiltonian has only {self.number_eta_params} parameters. But index {self.param_index} was requested"
                )
        else:
            self.number_eta_params = 0
            print(
                "Warning: the Hamiltonian has no VCN parameter, nothing will be measured"
            )

    def get_expectation_value(
        self, time: float, system_state: state.SystemState
    ) -> np.complex128 | np.ndarray:
        _ = time
        _ = system_state

        if self.can_return_value:
            etas = self.vcn_hamiltonian.eta_vec

            if self.param_real_part:
                return np.real(etas[self.param_index])
            else:
                return np.imag(etas[self.param_index])
        else:
            return 0

    def get_label(self) -> str:
        return f"{'Re' if self.param_real_part else 'Im'}-Part of VCN Parameter {self.param_index}"

    def get_log_info(self) -> Dict[str, Union[float, str, bool, Dict[Any, Any]]]:
        return {
            "type": "VCNFactor",
            "label": self.get_label(),
            "param_real_part": self.param_real_part,
            "param_index": self.param_index,
            "can_return_value": self.can_return_value,
        }
