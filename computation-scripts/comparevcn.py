import time
from time import time as measure
import state
import systemgeometry
import numpy as np
from randomgenerator import RandomGenerator
from variationalclassicalnetworks import (
    ChainDirectionDependentAllSameFirstOrder,
    SquareDirectionDependentAllSameFirstOrder,
)
from hamiltonian import (
    FirstOrderVariationalClassicalNetworkAnalyticalParamsHamiltonian,
)
import csv


def compare_arrays(src: str, comp_a: np.ndarray, comp_b: np.ndarray):
    if np.sum(np.abs(comp_a - comp_b)) > 1e-9:
        print(comp_a)
        print(comp_b)
        raise Exception(f"The arrays have differences: {src}")


U = 0.3
E = -0.5
J = 1.1
phi = np.pi / 3.1
measurement_time = 5 * (1 / J)


do_additional_checks = (
    True  # this makes it take more time for collecting data this should be disabled
)


random = RandomGenerator(str(time.time()))

iterations = 300
range_upper_bound = 226  # end is excluded
range_upper_bound_square = 16  # end is excluded
for geometry_type in ["chain", "square"]:

    fileo1 = open(
        f"./../run-outputs/{geometry_type}_measurements_vcno1.csv", mode="w", newline=""
    )
    writero1 = csv.writer(fileo1, lineterminator="\n")
    # Descriptor
    writero1.writerow(
        [
            f"# Timing measurements on the {geometry_type}-geometry for the first order perturbation theory"
        ]
    )
    # Headers
    header_array = [
        "n",
        "flip can",
        "flip opt",
        "double flip can",
        "double flip opt",
        "swap can",
        "swap opt",
    ]
    writero1.writerow(header_array)

    if geometry_type == "chain":
        use_range = range(5, range_upper_bound, 5)  # otherwise too many data points
    elif geometry_type == "square":
        use_range = range(3, range_upper_bound_square)
    else:
        raise Exception("Not supported geometry")
    for n in use_range:
        print(f"Doing n={n} and geometry {geometry_type}:")
        print()

        # Caution: This breaks from linear Chain, n<=3 because then no 3 different comparison indices can be found
        if geometry_type == "chain":
            system_geometry = systemgeometry.LinearChainNonPeriodicState(n)
        elif geometry_type == "square":
            system_geometry = systemgeometry.SquareSystemNonPeriodicState(n)
        else:
            raise Exception("Not supported geometry")

        initial_system_state = state.HomogenousInitialSystemState(system_geometry)

        use_state = state.SystemState(system_geometry, initial_system_state)

        if geometry_type == "chain":
            vcn_helper = ChainDirectionDependentAllSameFirstOrder(
                system_geometry=system_geometry, J=J
            )
        elif geometry_type == "square":
            vcn_helper = SquareDirectionDependentAllSameFirstOrder(
                system_geometry=system_geometry, J=J
            )
        else:
            raise Exception("Not supported geometry")

        ham = FirstOrderVariationalClassicalNetworkAnalyticalParamsHamiltonian(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            psi_selection=vcn_helper,
            random_generator=random,
            vcn_parameter_init_distribution="normal",
        )

        total_time_optimized_single = 0
        total_time_un_optimized_single = 0
        total_time_optimized_double = 0
        total_time_un_optimized_double = 0

        total_time_h_eff_direct_flipping = 0
        total_time_h_eff_optimized_flipping = 0

        total_time_h_eff_direct_double_flipping = 0
        total_time_h_eff_optimized_double_flipping = 0

        total_time_h_eff_direct_swapping = 0
        total_time_h_eff_optimized_swapping = 0

        for _ in range(iterations):
            use_state.init_random_filling(random)

            sw1_up: bool = random.randbool()
            sw1_index: int = random.randint(
                0, use_state.get_number_sites_wo_spin_degree() - 1
            )

            sw2_up: bool = random.randbool()
            sw2_index: int = random.randint(
                0, use_state.get_number_sites_wo_spin_degree() - 1
            )
            while sw2_index in [sw1_index]:
                sw2_index = random.randint(
                    0, use_state.get_number_sites_wo_spin_degree() - 1
                )

            ham.initialize(time=measurement_time)

            if do_additional_checks:
                a = measure() * 1000
                res_optimized = vcn_helper.eval_PSI_differences_flipping(
                    l=sw1_index, spin_up=sw1_up, before_swap_system_state=use_state
                )
                b = measure() * 1000
                res_un_optimized = vcn_helper.eval_PSI_differences_flipping_unoptimized(
                    l=sw1_index, spin_up=sw1_up, before_swap_system_state=use_state
                )
                c = measure() * 1000
                compare_arrays("single", res_optimized, res_un_optimized)
                total_time_optimized_single += b - a
                total_time_un_optimized_single += c - b

                a = measure() * 1000
                res_optimized_double = vcn_helper.eval_PSI_differences_double_flipping(
                    l=sw1_index,
                    m=sw2_index,
                    spin_l_up=sw1_up,
                    spin_m_up=sw2_up,
                    before_swap_system_state=use_state,
                )
                b = measure() * 1000
                res_un_optimized_double = (
                    vcn_helper.eval_PSI_differences_double_flipping_unoptimized(
                        l=sw1_index,
                        m=sw2_index,
                        spin_l_up=sw1_up,
                        spin_m_up=sw2_up,
                        before_swap_system_state=use_state,
                    )
                )
                c = measure() * 1000
                compare_arrays("double", res_optimized_double, res_un_optimized_double)
                total_time_optimized_double += b - a
                total_time_un_optimized_double += c - b

            a = measure() * 1000
            res_heff_flipping_optimized = ham.get_H_eff_difference_flipping(
                time=measurement_time,
                flipping_up=sw1_up,
                flipping_index=sw1_index,
                before_swap_system_state=use_state,
            )[0]
            copy = use_state.get_editable_copy()
            copy.flip_in_place(flipping_up=sw1_up, flipping_index=sw1_index)
            b = measure() * 1000
            res_heff_flipping_direct_A = ham.get_H_eff(
                time=measurement_time,
                system_state=use_state,
            )
            res_heff_flipping_direct_B = ham.get_H_eff(
                time=measurement_time,
                system_state=copy,
            )
            res_heff_flipping_direct = (
                res_heff_flipping_direct_A - res_heff_flipping_direct_B
            )
            c = measure() * 1000
            compare_arrays(
                "heff flipping",
                np.array([res_heff_flipping_optimized]),
                np.array([res_heff_flipping_direct]),
            )
            total_time_h_eff_optimized_flipping += b - a
            total_time_h_eff_direct_flipping += c - b

            a = measure() * 1000
            res_heff_double_flipping_optimized = (
                ham.get_H_eff_difference_double_flipping(
                    time=measurement_time,
                    flipping1_up=sw1_up,
                    flipping1_index=sw1_index,
                    flipping2_up=sw2_up,
                    flipping2_index=sw2_index,
                    before_swap_system_state=use_state,
                )[0]
            )
            copy = use_state.get_editable_copy()
            copy.flip_in_place(flipping_up=sw1_up, flipping_index=sw1_index)
            copy.flip_in_place(flipping_up=sw2_up, flipping_index=sw2_index)
            b = measure() * 1000
            res_heff_double_flipping_direct_A = ham.get_H_eff(
                time=measurement_time,
                system_state=use_state,
            )
            res_heff_double_flipping_direct_B = ham.get_H_eff(
                time=measurement_time,
                system_state=copy,
            )
            res_heff_double_flipping_direct = (
                res_heff_double_flipping_direct_A - res_heff_double_flipping_direct_B
            )
            c = measure() * 1000
            compare_arrays(
                "heff double flipping",
                np.array([res_heff_double_flipping_optimized]),
                np.array([res_heff_double_flipping_direct]),
            )
            total_time_h_eff_optimized_double_flipping += b - a
            total_time_h_eff_direct_double_flipping += c - b

            a = measure() * 1000
            res_heff_swapping_optimized = ham.get_H_eff_difference_swapping(
                time=measurement_time,
                sw1_index=sw1_index,
                sw2_index=sw2_index,
                sw1_up=sw1_up,
                sw2_up=sw2_up,
                before_swap_system_state=use_state,
            )[0]
            copy = use_state.get_editable_copy()
            copy.swap_in_place(
                sw1_index=sw1_index,
                sw2_index=sw2_index,
                sw1_up=sw1_up,
                sw2_up=sw2_up,
            )
            b = measure() * 1000
            res_heff_swapping_direct_A = ham.get_H_eff(
                time=measurement_time,
                system_state=use_state,
            )
            res_heff_swapping_direct_B = ham.get_H_eff(
                time=measurement_time,
                system_state=copy,
            )
            res_heff_swapping_direct = (
                res_heff_swapping_direct_A - res_heff_swapping_direct_B
            )
            c = measure() * 1000
            compare_arrays(
                "heff swapping",
                np.array([res_heff_swapping_optimized]),
                np.array([res_heff_swapping_direct]),
            )
            total_time_h_eff_optimized_swapping += b - a
            total_time_h_eff_direct_swapping += c - b

        print("Single Flipping")
        print(f"total time un-optimized flipping: {total_time_un_optimized_single}")
        print(f"total time optimized flipping: {total_time_optimized_single}")

        print()
        print("Double Flipping")
        print(f"total time un-optimized flipping: {total_time_un_optimized_double}")
        print(f"total time optimized flipping: {total_time_optimized_double}")

        print("Heff Single Flipping")
        print(
            f"Heff total time un-optimized flipping: {total_time_h_eff_direct_flipping}"
        )
        print(
            f"Heff total time optimized flipping: {total_time_h_eff_optimized_flipping}"
        )

        print("Heff Double Flipping")
        print(
            f"Heff total time un-optimized double flipping: {total_time_h_eff_direct_double_flipping}"
        )
        print(
            f"Heff total time optimized double flipping: {total_time_h_eff_optimized_double_flipping}"
        )

        print("Heff Swapping")
        print(
            f"Heff total time un-optimized swapping: {total_time_h_eff_direct_swapping}"
        )
        print(
            f"Heff total time optimized swapping: {total_time_h_eff_optimized_swapping}"
        )

        writero1.writerow(
            [
                n,
                total_time_h_eff_direct_flipping,
                total_time_h_eff_optimized_flipping,
                total_time_h_eff_direct_double_flipping,
                total_time_h_eff_optimized_double_flipping,
                total_time_h_eff_direct_swapping,
                total_time_h_eff_optimized_swapping,
            ]
        )

    fileo1.close()
