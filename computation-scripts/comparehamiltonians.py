import time
import state
import systemgeometry
import numpy as np
import hamiltonian
from randomgenerator import RandomGenerator
from time import time as measure
import csv

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
range_upper_bound_square = 15  # end is excluded
for geometry_type in ["chain", "square"]:

    fileo1 = open(
        f"./../run-outputs/{geometry_type}_measurements_o1.csv", mode="w", newline=""
    )
    fileo2 = open(
        f"./../run-outputs/{geometry_type}_measurements_o2.csv", mode="w", newline=""
    )
    writero1 = csv.writer(fileo1, lineterminator="\n")
    writero2 = csv.writer(fileo2, lineterminator="\n")
    # Descriptor
    writero1.writerow(
        [
            f"# Timing measurements on the {geometry_type}-geometry for the first order perturbation theory"
        ]
    )
    writero2.writerow(
        [
            f"# Timing measurements on the {geometry_type}-geometry for the second order perturbation theory"
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
    writero2.writerow(header_array)

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

        ham_canonical_legacy = hamiltonian.FirstOrderDifferentiatesPsiHamiltonian(
            U=U, E=E, J=J, phi=phi
        )
        ham_canonical = hamiltonian.FirstOrderCanonicalHamiltonian(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
        ham_swap_optimized = hamiltonian.FirstOrderSwappingOptimizedHamiltonian(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
        ham_flip_optimized = hamiltonian.FirstOrderFlippingOptimizedHamiltonian(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
        ham_canonical_second_order = hamiltonian.SecondOrderCanonicalHamiltonian(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            system_geometry=system_geometry,
        )
        ham_second_order_optimized = hamiltonian.SecondOrderOptimizedHamiltonian(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            system_geometry=system_geometry,
        )

        def compare_arrays(comp_a: np.ndarray, comp_b: np.ndarray):
            if np.sum(np.abs(comp_a - comp_b)) > 1e-9:
                raise Exception("Termination because arrays have been changed")

        use_state = state.SystemState(system_geometry, initial_system_state)

        total_time_new = 0
        total_time_legacy = 0

        total_time_swapping_un_optimized = 0
        total_time_swapping_optimized = 0

        total_time_flipping_un_optimized = 0
        total_time_flipping_optimized = 0

        total_time_double_flipping_un_optimized = 0
        total_time_double_flipping_optimized = 0

        total_time_swapping_un_optimized_second_order = 0
        total_time_swapping_optimized_second_order = 0

        total_time_flipping_un_optimized_second_order = 0
        total_time_flipping_optimized_second_order = 0

        total_time_double_flipping_un_optimized_second_order = 0
        total_time_double_flipping_optimized_second_order = 0

        for _ in range(iterations):
            use_state.init_random_filling(random)

            sw1_up: bool = random.randbool()
            sw1_index: int = random.randint(
                0, use_state.get_number_sites_wo_spin_degree() - 1
            )
            sw1_index_neighbors = system_geometry.get_nearest_neighbor_indices(
                sw1_index
            )
            sw2_up: bool = random.randbool()
            sw2_index: int = sw1_index_neighbors[0]
            sw3_up: bool = random.randbool()
            sw3_index: int = sw2_index
            while sw3_index in sw1_index_neighbors + [sw1_index]:
                sw3_index: int = random.randint(
                    0, use_state.get_number_sites_wo_spin_degree() - 1
                )

            # start manually modifying array copies for external comparison
            direct_access_sw1_index = (
                sw1_index + (not sw1_up) * use_state.get_number_sites_wo_spin_degree()
            )
            direct_access_sw2_index = (
                sw2_index + (not sw2_up) * use_state.get_number_sites_wo_spin_degree()
            )
            direct_access_same_site_index = (  # special test, both sw1
                sw1_index + (not sw2_up) * use_state.get_number_sites_wo_spin_degree()
            )
            direct_access_sw3_index = (
                sw3_index + (not sw3_up) * use_state.get_number_sites_wo_spin_degree()
            )

            samesite_double_flipped_copy = use_state.get_editable_copy()
            samesite_double_flipped_copy.get_state_array()[direct_access_sw1_index] = (
                1
                - samesite_double_flipped_copy.get_state_array()[
                    direct_access_sw1_index
                ]
            )
            samesite_double_flipped_copy.get_state_array()[
                direct_access_same_site_index
            ] = (
                1
                - samesite_double_flipped_copy.get_state_array()[
                    direct_access_same_site_index
                ]
            )

            adjacent_double_flipped_copy = use_state.get_editable_copy()
            adjacent_double_flipped_copy.get_state_array()[direct_access_sw1_index] = (
                1
                - adjacent_double_flipped_copy.get_state_array()[
                    direct_access_sw1_index
                ]
            )
            adjacent_double_flipped_copy.get_state_array()[direct_access_sw2_index] = (
                1
                - adjacent_double_flipped_copy.get_state_array()[
                    direct_access_sw2_index
                ]
            )

            far_double_flipped_copy = use_state.get_editable_copy()
            far_double_flipped_copy.get_state_array()[direct_access_sw1_index] = (
                1 - far_double_flipped_copy.get_state_array()[direct_access_sw1_index]
            )
            far_double_flipped_copy.get_state_array()[direct_access_sw3_index] = (
                1 - far_double_flipped_copy.get_state_array()[direct_access_sw3_index]
            )
            # end manually modifying array copies for external comparison

            if sw2_index not in system_geometry.get_nearest_neighbor_indices(sw1_index):
                raise Exception("Expect to be neighbor")
            if sw3_index in system_geometry.get_nearest_neighbor_indices(sw1_index):
                raise Exception("Expected not to be a neighbor")

            modification_check_comparearray = np.copy(use_state.get_state_array())

            if do_additional_checks:
                # check spin-symmetry by swapping the occupation array and with that checking symmetry between up and down
                spin_inverted_copy = use_state.get_editable_copy()
                size_of_system = use_state.get_number_sites_wo_spin_degree()
                for site in range(size_of_system):
                    arr = spin_inverted_copy.get_state_array()
                    a = arr[site]
                    arr[site] = arr[site + size_of_system]
                    arr[site + size_of_system] = a
                non_inv = ham_canonical.get_H_eff(
                    time=measurement_time,
                    system_state=use_state,
                )
                compare_arrays(
                    modification_check_comparearray, use_state.get_state_array()
                )
                inv = ham_canonical.get_H_eff(
                    time=measurement_time,
                    system_state=spin_inverted_copy,
                )
                if np.abs(non_inv - inv) > 1e-6:
                    print("Difference after spin inversion")
                    print(use_state.get_state_array())
                    print(non_inv)
                    print(inv)
                    raise Exception("Difference Error above")

                non_inv_sw = ham_canonical.get_H_eff_difference_swapping(
                    time=measurement_time,
                    sw1_up=sw1_up,
                    sw1_index=sw1_index,
                    sw2_up=sw2_up,
                    sw2_index=sw2_index,
                    before_swap_system_state=use_state,
                )[0]
                compare_arrays(
                    modification_check_comparearray, use_state.get_state_array()
                )
                inv_sw = ham_canonical.get_H_eff_difference_swapping(
                    time=measurement_time,
                    sw1_up=not sw1_up,
                    sw1_index=sw1_index,
                    sw2_up=not sw2_up,
                    sw2_index=sw2_index,
                    before_swap_system_state=spin_inverted_copy,
                )[0]
                if np.abs(non_inv_sw - inv_sw) > 1e-6:
                    print("Difference of swapping after spin inversion")
                    print(use_state.get_state_array())
                    print(non_inv_sw)
                    print(inv_sw)
                    raise Exception("Difference Error above")

                # check end-to-end for optimization
                canonical_e2e = ham_canonical_second_order.get_H_eff(
                    time=measurement_time,
                    system_state=use_state,
                )
                compare_arrays(
                    modification_check_comparearray, use_state.get_state_array()
                )
                optimized_e2e = ham_second_order_optimized.get_H_eff(
                    time=measurement_time,
                    system_state=spin_inverted_copy,
                )
                if np.abs(canonical_e2e - optimized_e2e) > 1e-6:
                    print("Difference end-to-end")
                    print(use_state.get_state_array())
                    print(canonical_e2e)
                    print(optimized_e2e)
                    raise Exception("Difference Error above")

            # check simplifications

            a = measure() * 1000
            res_new = ham_canonical.get_H_eff(
                time=measurement_time,
                system_state=use_state,
            )
            b = measure() * 1000
            res_legacy = ham_canonical_legacy.get_H_eff(
                time=measurement_time,
                system_state=use_state,
            )
            c = measure() * 1000
            compare_arrays(modification_check_comparearray, use_state.get_state_array())
            total_time_new += b - a
            total_time_legacy += c - b
            if np.abs(res_new - res_legacy) > 1e-6:
                print("Difference for New implementation")
                print(use_state.get_state_array())
                print(res_new)
                print(res_legacy)
                raise Exception("Difference Error above")

            a = measure() * 1000
            res_a = ham_canonical.get_H_eff_difference_swapping(
                time=measurement_time,
                sw1_up=sw1_up,
                sw1_index=sw1_index,
                sw2_up=sw2_up,
                sw2_index=sw2_index,
                before_swap_system_state=use_state,
            )
            b = measure() * 1000
            res_b = ham_swap_optimized.get_H_eff_difference_swapping(
                time=measurement_time,
                sw1_up=sw1_up,
                sw1_index=sw1_index,
                sw2_up=sw2_up,
                sw2_index=sw2_index,
                before_swap_system_state=use_state,
            )
            c = measure() * 1000
            compare_arrays(modification_check_comparearray, use_state.get_state_array())
            total_time_swapping_un_optimized += b - a
            total_time_swapping_optimized += c - b
            if np.abs(res_a[0] - res_b[0]) > 1e-6:
                print("Difference for Hopping")
                print(res_a[0])
                print(res_b[0])
                raise Exception("Difference Error above")

            a = measure() * 1000
            res_a_far = ham_canonical.get_H_eff_difference_swapping(
                time=measurement_time,
                sw1_up=sw1_up,
                sw1_index=sw1_index,
                sw2_up=sw3_up,
                sw2_index=sw3_index,
                before_swap_system_state=use_state,
            )
            b = measure() * 1000
            res_b_far = ham_swap_optimized.get_H_eff_difference_swapping(
                time=measurement_time,
                sw1_up=sw1_up,
                sw1_index=sw1_index,
                sw2_up=sw3_up,
                sw2_index=sw3_index,
                before_swap_system_state=use_state,
            )
            c = measure() * 1000
            compare_arrays(modification_check_comparearray, use_state.get_state_array())
            total_time_swapping_un_optimized += b - a
            total_time_swapping_optimized += c - b
            if np.abs(res_a_far[0] - res_b_far[0]) > 1e-6:
                print("Difference for far Hopping")
                print(res_a_far[0])
                print(res_b_far[0])
                raise Exception("Difference Error above")

            a = measure() * 1000
            res_flip_a = ham_canonical.get_H_eff_difference_flipping(
                time=measurement_time,
                flipping_up=sw1_up,
                flipping_index=sw1_index,
                before_swap_system_state=use_state,
            )
            b = measure() * 1000
            res_flip_b = ham_flip_optimized.get_H_eff_difference_flipping(
                time=measurement_time,
                flipping_up=sw1_up,
                flipping_index=sw1_index,
                before_swap_system_state=use_state,
            )
            c = measure() * 1000
            compare_arrays(modification_check_comparearray, use_state.get_state_array())
            total_time_flipping_un_optimized += b - a
            total_time_flipping_optimized += c - b
            if np.abs(res_flip_a[0] - res_flip_b[0]) > 1e-6:
                print("Difference for Flipping")
                print(res_flip_a[0])
                print(res_flip_b[0])
                raise Exception("Difference Error above")

            a = measure() * 1000
            res_double_flip_a_same = ham_canonical.get_H_eff_difference_double_flipping(
                time=measurement_time,
                flipping1_up=sw1_up,
                flipping1_index=sw1_index,  # same site!!
                flipping2_up=sw2_up,
                flipping2_index=sw1_index,  # same site!!
                before_swap_system_state=use_state,
            )
            b = measure() * 1000
            res_double_flip_b_same = (
                ham_flip_optimized.get_H_eff_difference_double_flipping(
                    time=measurement_time,
                    flipping1_up=sw1_up,
                    flipping1_index=sw1_index,  # same site!!
                    flipping2_up=sw2_up,
                    flipping2_index=sw1_index,  # same site!!
                    before_swap_system_state=use_state,
                )
            )
            c = measure() * 1000
            compare_arrays(modification_check_comparearray, use_state.get_state_array())
            total_time_double_flipping_un_optimized += b - a
            total_time_double_flipping_optimized += c - b
            if np.abs(res_double_flip_a_same[0] - res_double_flip_b_same[0]) > 1e-6:
                print("Difference for Double Flipping Same Site")
                print(res_double_flip_a_same[0])
                print(res_double_flip_b_same[0])
                raise Exception("Difference Error above")

            val_from_manual_caculation = ham_canonical.get_H_eff(
                time=measurement_time, system_state=use_state
            ) - ham_canonical.get_H_eff(
                time=measurement_time, system_state=samesite_double_flipped_copy
            )
            if np.abs(res_double_flip_a_same[0] - val_from_manual_caculation) > 1e-6:
                print("Difference for Double Flipping Same Site with manual")
                print(val_from_manual_caculation)
                print(res_double_flip_a_same[0])
                raise Exception("Difference Error above")

            a = measure() * 1000
            res_double_flip_a = ham_canonical.get_H_eff_difference_double_flipping(
                time=measurement_time,
                flipping1_up=sw1_up,
                flipping1_index=sw1_index,
                flipping2_up=sw2_up,
                flipping2_index=sw2_index,
                before_swap_system_state=use_state,
            )
            b = measure() * 1000
            res_double_flip_b = ham_flip_optimized.get_H_eff_difference_double_flipping(
                time=measurement_time,
                flipping1_up=sw1_up,
                flipping1_index=sw1_index,
                flipping2_up=sw2_up,
                flipping2_index=sw2_index,
                before_swap_system_state=use_state,
            )
            c = measure() * 1000
            compare_arrays(modification_check_comparearray, use_state.get_state_array())
            total_time_double_flipping_un_optimized += b - a
            total_time_double_flipping_optimized += c - b
            if np.abs(res_double_flip_a[0] - res_double_flip_b[0]) > 1e-6:
                print("Difference for Double Flipping")
                print(res_double_flip_a[0])
                print(res_double_flip_b[0])
                raise Exception("Difference Error above")

            val_from_manual_caculation = ham_canonical.get_H_eff(
                time=measurement_time, system_state=use_state
            ) - ham_canonical.get_H_eff(
                time=measurement_time, system_state=adjacent_double_flipped_copy
            )
            if np.abs(res_double_flip_a[0] - val_from_manual_caculation) > 1e-6:
                print("Difference for Double Flipping with manual")
                print(val_from_manual_caculation)
                print(res_double_flip_a[0])
                raise Exception("Difference Error above")

            a = measure() * 1000
            res_double_flip_a_far = ham_canonical.get_H_eff_difference_double_flipping(
                time=measurement_time,
                flipping1_up=sw1_up,
                flipping1_index=sw1_index,
                flipping2_up=sw3_up,
                flipping2_index=sw3_index,
                before_swap_system_state=use_state,
            )
            b = measure() * 1000
            res_double_flip_b_far = (
                ham_flip_optimized.get_H_eff_difference_double_flipping(
                    time=measurement_time,
                    flipping1_up=sw1_up,
                    flipping1_index=sw1_index,
                    flipping2_up=sw3_up,
                    flipping2_index=sw3_index,
                    before_swap_system_state=use_state,
                )
            )
            c = measure() * 1000
            compare_arrays(modification_check_comparearray, use_state.get_state_array())
            total_time_double_flipping_un_optimized += b - a
            total_time_double_flipping_optimized += c - b
            if np.abs(res_double_flip_a_far[0] - res_double_flip_b_far[0]) > 1e-6:
                print("Difference for far Double Flipping")
                print(res_double_flip_a_far[0])
                print(res_double_flip_b_far[0])
                raise Exception("Difference Error above")

            val_from_manual_caculation = ham_canonical.get_H_eff(
                time=measurement_time, system_state=use_state
            ) - ham_canonical.get_H_eff(
                time=measurement_time, system_state=far_double_flipped_copy
            )
            if np.abs(res_double_flip_a_far[0] - val_from_manual_caculation) > 1e-6:
                print("Difference for far Double Flipping with manual")
                print(val_from_manual_caculation)
                print(res_double_flip_a_far[0])
                raise Exception("Difference Error above")

            a = measure() * 1000
            res_a = ham_canonical_second_order.get_H_eff_difference_swapping(
                time=measurement_time,
                sw1_up=sw1_up,
                sw1_index=sw1_index,
                sw2_up=sw2_up,
                sw2_index=sw2_index,
                before_swap_system_state=use_state,
            )
            b = measure() * 1000
            res_b = ham_second_order_optimized.get_H_eff_difference_swapping(
                time=measurement_time,
                sw1_up=sw1_up,
                sw1_index=sw1_index,
                sw2_up=sw2_up,
                sw2_index=sw2_index,
                before_swap_system_state=use_state,
            )
            c = measure() * 1000
            compare_arrays(modification_check_comparearray, use_state.get_state_array())
            total_time_swapping_un_optimized_second_order += b - a
            total_time_swapping_optimized_second_order += c - b
            if np.abs(res_a[0] - res_b[0]) > 1e-6:
                print("Second Order Difference for Hopping")
                print(res_a[0])
                print(res_b[0])
                raise Exception("Difference Error above")

            a = measure() * 1000
            res_a_far = ham_canonical_second_order.get_H_eff_difference_swapping(
                time=measurement_time,
                sw1_up=sw1_up,
                sw1_index=sw1_index,
                sw2_up=sw3_up,
                sw2_index=sw3_index,
                before_swap_system_state=use_state,
            )
            b = measure() * 1000
            res_b_far = ham_second_order_optimized.get_H_eff_difference_swapping(
                time=measurement_time,
                sw1_up=sw1_up,
                sw1_index=sw1_index,
                sw2_up=sw3_up,
                sw2_index=sw3_index,
                before_swap_system_state=use_state,
            )
            c = measure() * 1000
            compare_arrays(modification_check_comparearray, use_state.get_state_array())
            total_time_swapping_un_optimized_second_order += b - a
            total_time_swapping_optimized_second_order += c - b
            if np.abs(res_a_far[0] - res_b_far[0]) > 1e-6:
                print("Second Order Difference for far Hopping")
                print(res_a_far[0])
                print(res_b_far[0])
                raise Exception("Difference Error above")

            a = measure() * 1000
            res_flip_a = ham_canonical_second_order.get_H_eff_difference_flipping(
                time=measurement_time,
                flipping_up=sw1_up,
                flipping_index=sw1_index,
                before_swap_system_state=use_state,
            )
            b = measure() * 1000
            res_flip_b = ham_second_order_optimized.get_H_eff_difference_flipping(
                time=measurement_time,
                flipping_up=sw1_up,
                flipping_index=sw1_index,
                before_swap_system_state=use_state,
            )
            c = measure() * 1000
            compare_arrays(modification_check_comparearray, use_state.get_state_array())
            total_time_flipping_un_optimized_second_order += b - a
            total_time_flipping_optimized_second_order += c - b
            if np.abs(res_flip_a[0] - res_flip_b[0]) > 1e-6:
                print("Second Order Difference for Flipping")
                print(res_flip_a[0])
                print(res_flip_b[0])
                raise Exception("Difference Error above")

            a = measure() * 1000
            res_double_flip_a_same = (
                ham_canonical_second_order.get_H_eff_difference_double_flipping(
                    time=measurement_time,
                    flipping1_up=sw1_up,
                    flipping1_index=sw1_index,  # same site!!
                    flipping2_up=sw2_up,
                    flipping2_index=sw1_index,  # same site!!
                    before_swap_system_state=use_state,
                )
            )
            b = measure() * 1000
            res_double_flip_b_same = (
                ham_second_order_optimized.get_H_eff_difference_double_flipping(
                    time=measurement_time,
                    flipping1_up=sw1_up,
                    flipping1_index=sw1_index,  # same site!!
                    flipping2_up=sw2_up,
                    flipping2_index=sw1_index,  # same site!!
                    before_swap_system_state=use_state,
                )
            )
            c = measure() * 1000
            compare_arrays(modification_check_comparearray, use_state.get_state_array())
            total_time_double_flipping_un_optimized_second_order += b - a
            total_time_double_flipping_optimized_second_order += c - b
            if np.abs(res_double_flip_a_same[0] - res_double_flip_b_same[0]) > 1e-6:
                print("Second Order Difference for Double Flipping Same Site")
                print(res_double_flip_a_same[0])
                print(res_double_flip_b_same[0])
                raise Exception("Difference Error above")

            a = measure() * 1000
            res_double_flip_a = (
                ham_canonical_second_order.get_H_eff_difference_double_flipping(
                    time=measurement_time,
                    flipping1_up=sw1_up,
                    flipping1_index=sw1_index,
                    flipping2_up=sw2_up,
                    flipping2_index=sw2_index,
                    before_swap_system_state=use_state,
                )
            )
            b = measure() * 1000
            res_double_flip_b = (
                ham_second_order_optimized.get_H_eff_difference_double_flipping(
                    time=measurement_time,
                    flipping1_up=sw1_up,
                    flipping1_index=sw1_index,
                    flipping2_up=sw2_up,
                    flipping2_index=sw2_index,
                    before_swap_system_state=use_state,
                )
            )
            c = measure() * 1000
            compare_arrays(modification_check_comparearray, use_state.get_state_array())
            total_time_double_flipping_un_optimized_second_order += b - a
            total_time_double_flipping_optimized_second_order += c - b
            if np.abs(res_double_flip_a[0] - res_double_flip_b[0]) > 1e-6:
                print("Second Order Difference for Double Flipping")
                print(res_double_flip_a[0])
                print(res_double_flip_b[0])
                raise Exception("Difference Error above")

            a = measure() * 1000
            res_double_flip_a_far = (
                ham_canonical_second_order.get_H_eff_difference_double_flipping(
                    time=measurement_time,
                    flipping1_up=sw1_up,
                    flipping1_index=sw1_index,
                    flipping2_up=sw3_up,
                    flipping2_index=sw3_index,
                    before_swap_system_state=use_state,
                )
            )
            b = measure() * 1000
            res_double_flip_b_far = (
                ham_second_order_optimized.get_H_eff_difference_double_flipping(
                    time=measurement_time,
                    flipping1_up=sw1_up,
                    flipping1_index=sw1_index,
                    flipping2_up=sw3_up,
                    flipping2_index=sw3_index,
                    before_swap_system_state=use_state,
                )
            )
            c = measure() * 1000
            compare_arrays(modification_check_comparearray, use_state.get_state_array())
            total_time_double_flipping_un_optimized_second_order += b - a
            total_time_double_flipping_optimized_second_order += c - b
            if np.abs(res_double_flip_a_far[0] - res_double_flip_b_far[0]) > 1e-6:
                print("Second Order Difference for far Double Flipping")
                print(res_double_flip_a_far[0])
                print(res_double_flip_b_far[0])
                raise Exception("Difference Error above")

        print("legacy ms:", total_time_legacy)
        print("new ms:", total_time_new)

        print()

        print("hopping ms:", total_time_swapping_un_optimized)
        print("hopping optimized ms:", total_time_swapping_optimized)

        print()

        print("flipping ms:", total_time_flipping_un_optimized)
        print("flipping optimized ms:", total_time_flipping_optimized)

        print()

        print("double flipping ms:", total_time_double_flipping_un_optimized)
        print("double flipping optimized ms:", total_time_double_flipping_optimized)

        print()

        print(
            "SECOND ORDER: hopping ms:", total_time_swapping_un_optimized_second_order
        )
        print(
            "SECOND ORDER: hopping optimized ms:",
            total_time_swapping_optimized_second_order,
        )

        print()

        print(
            "SECOND ORDER: flipping ms:", total_time_flipping_un_optimized_second_order
        )
        print(
            "SECOND ORDER: flipping optimized ms:",
            total_time_flipping_optimized_second_order,
        )

        print()

        print(
            "SECOND ORDER: double flipping ms:",
            total_time_double_flipping_un_optimized_second_order,
        )
        print(
            "SECOND ORDER: double flipping optimized ms:",
            total_time_double_flipping_optimized_second_order,
        )

        print()

        print(
            "Verified the correctness for the timing optimizations for:",
            iterations,
            " cases",
        )

        # "n",
        # "flip can",
        # "flip opt",
        # "double flip can",
        # "double flip opt",
        # "swap can",
        # "swap opt",

        # write the rows
        writero1.writerow(
            [
                n,
                total_time_flipping_un_optimized,
                total_time_flipping_optimized,
                total_time_double_flipping_un_optimized,
                total_time_double_flipping_optimized,
                total_time_swapping_un_optimized,
                total_time_swapping_optimized,
            ]
        )
        writero2.writerow(
            [
                n,
                total_time_flipping_un_optimized_second_order,
                total_time_flipping_optimized_second_order,
                total_time_double_flipping_un_optimized_second_order,
                total_time_double_flipping_optimized_second_order,
                total_time_swapping_un_optimized_second_order,
                total_time_swapping_optimized_second_order,
            ]
        )

    # cleanup after a geometry
    fileo1.close()
    fileo2.close()

if do_additional_checks:
    # !! Symmetry checks for electrical field inversion. Only chain and no timing

    system_geometry_chain = systemgeometry.LinearChainNonPeriodicState(3)
    initial_system_state_chain = state.HomogenousInitialSystemState(system_geometry)
    use_state_chain = state.SystemState(system_geometry_chain, initial_system_state)
    hamiltonian_chain_plus = hamiltonian.SecondOrderCanonicalHamiltonian(
        U=U,
        E=E,
        J=J,
        phi=phi,
        initial_system_state=initial_system_state_chain,
        system_geometry=system_geometry_chain,
    )
    hamiltonian_chain_minus = hamiltonian.SecondOrderCanonicalHamiltonian(
        U=U,
        E=-E,
        J=J,
        phi=phi,
        initial_system_state=initial_system_state_chain,
        system_geometry=system_geometry_chain,
    )

    iterations_symmetry = 10
    for _ in range(iterations_symmetry):
        use_state_chain.init_random_filling(random)

        # check locational symmetry, by flipping the occupation array
        locationally_inverted_copy = use_state_chain.get_editable_copy()
        size_of_system = use_state_chain.get_number_sites_wo_spin_degree()
        for site in range(size_of_system // 2):
            arr = locationally_inverted_copy.get_state_array()
            first_index = site
            second_index = size_of_system - site - 1

            # locationally swap up
            a = arr[first_index]
            arr[first_index] = arr[second_index]
            arr[second_index] = a

            # locationally swap down
            a = arr[first_index + size_of_system]
            arr[first_index + size_of_system] = arr[second_index + size_of_system]
            arr[second_index + size_of_system] = a

        index_to_flip = random.randint(0, size_of_system - 1)
        index_to_flip_inverse = size_of_system - index_to_flip - 1
        flip_up: bool = random.randbool()

        plus_energy_gap = hamiltonian_chain_plus.get_H_eff_difference_flipping(
            time=measurement_time,
            flipping_up=flip_up,
            flipping_index=index_to_flip,
            before_swap_system_state=use_state_chain,
        )[0]
        minus_energy_gap = hamiltonian_chain_minus.get_H_eff_difference_flipping(
            time=measurement_time,
            flipping_up=flip_up,
            flipping_index=index_to_flip_inverse,
            before_swap_system_state=locationally_inverted_copy,
        )[0]

        if np.abs(plus_energy_gap - minus_energy_gap) > 1e-6:
            print("Locational Swap difference gap")
            print(
                flip_up,
                index_to_flip,
                index_to_flip_inverse,
                use_state_chain.get_state_array(),
                locationally_inverted_copy.get_state_array(),
            )
            print(plus_energy_gap)
            print(minus_energy_gap)
            raise Exception("Difference Error above")

        plus_energy = hamiltonian_chain_plus.get_H_eff(
            time=measurement_time,
            system_state=use_state_chain,
        )
        minus_energy = hamiltonian_chain_minus.get_H_eff(
            time=measurement_time,
            system_state=locationally_inverted_copy,
        )

        if np.abs(plus_energy - minus_energy) > 1e-6:
            print("Locational Swap difference")
            print(
                use_state_chain.get_state_array(),
                locationally_inverted_copy.get_state_array(),
            )
            print(plus_energy)
            print(minus_energy)
            raise Exception("Difference Error above")

    print(
        f"Verified the correctness for the locational inversion for {iterations_symmetry} iterations"
    )
