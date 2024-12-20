import time
import state
import systemgeometry
import numpy as np
import hamiltonian
from randomgenerator import RandomGenerator
from time import time as measure

U = 0.3
E = -0.5
J = 1.1
phi = np.pi / 3.1
measurement_time = 5 * (1 / J)

random = RandomGenerator(str(time.time()))

# Caution: This breaks from linear Chain, n<=3 because then no 3 different comparison indices can be found
system_geometry = systemgeometry.SquareSystemNonPeriodicState(5)

initial_system_state = state.HomogenousInitialSystemState(system_geometry)

ham_canonical_legacy = (
    hamiltonian.HardcoreBosonicHamiltonianStraightCalcPsiDiffFirstOrder(
        U=U, E=E, J=J, phi=phi
    )
)
ham_canonical = hamiltonian.HardcoreBosonicHamiltonian(
    U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
)
ham_swap_optimized = hamiltonian.HardcoreBosonicHamiltonianSwappingOptimization(
    U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
)
ham_flip_optimized = hamiltonian.HardcoreBosonicHamiltonianFlippingOptimization(
    U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
)
ham_canonical_second_order = hamiltonian.HardcoreBosonicHamiltonianSecondOrder(
    U=U,
    E=E,
    J=J,
    phi=phi,
    initial_system_state=initial_system_state,
    system_geometry=system_geometry,
)
ham_second_order_optimized = (
    hamiltonian.HardcoreBosonicHamiltonianFlippingAndSwappingOptimizationSecondOrder(
        U=U,
        E=E,
        J=J,
        phi=phi,
        initial_system_state=initial_system_state,
        system_geometry=system_geometry,
    )
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

iterations = 100
for _ in range(iterations):
    use_state.init_random_filling(random)

    sw1_up: bool = random.randbool()
    sw1_index: int = random.randint(0, use_state.get_number_sites_wo_spin_degree() - 1)
    sw1_index_neighbors = system_geometry.get_nearest_neighbor_indices(sw1_index)
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
        1 - samesite_double_flipped_copy.get_state_array()[direct_access_sw1_index]
    )
    samesite_double_flipped_copy.get_state_array()[direct_access_same_site_index] = (
        1
        - samesite_double_flipped_copy.get_state_array()[direct_access_same_site_index]
    )

    adjacent_double_flipped_copy = use_state.get_editable_copy()
    adjacent_double_flipped_copy.get_state_array()[direct_access_sw1_index] = (
        1 - adjacent_double_flipped_copy.get_state_array()[direct_access_sw1_index]
    )
    adjacent_double_flipped_copy.get_state_array()[direct_access_sw2_index] = (
        1 - adjacent_double_flipped_copy.get_state_array()[direct_access_sw2_index]
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
    compare_arrays(modification_check_comparearray, use_state.get_state_array())
    inv = ham_canonical.get_H_eff(
        time=measurement_time,
        system_state=spin_inverted_copy,
    )
    if np.abs(non_inv - inv) > 1e-6:
        print("Difference after spin inversion")
        print(use_state.get_state_array())
        print(non_inv)
        print(inv)
    non_inv_sw = ham_canonical.get_H_eff_difference_swapping(
        time=measurement_time,
        sw1_up=sw1_up,
        sw1_index=sw1_index,
        sw2_up=sw2_up,
        sw2_index=sw2_index,
        before_swap_system_state=use_state,
    )[0]
    compare_arrays(modification_check_comparearray, use_state.get_state_array())
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

    # check end-to-end for optimization
    canonical_e2e = ham_canonical_second_order.get_H_eff(
        time=measurement_time,
        system_state=use_state,
    )
    compare_arrays(modification_check_comparearray, use_state.get_state_array())
    optimized_e2e = ham_second_order_optimized.get_H_eff(
        time=measurement_time,
        system_state=spin_inverted_copy,
    )
    if np.abs(canonical_e2e - optimized_e2e) > 1e-6:
        print("Difference end-to-end")
        print(use_state.get_state_array())
        print(canonical_e2e)
        print(optimized_e2e)

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
    res_double_flip_b_same = ham_flip_optimized.get_H_eff_difference_double_flipping(
        time=measurement_time,
        flipping1_up=sw1_up,
        flipping1_index=sw1_index,  # same site!!
        flipping2_up=sw2_up,
        flipping2_index=sw1_index,  # same site!!
        before_swap_system_state=use_state,
    )
    c = measure() * 1000
    compare_arrays(modification_check_comparearray, use_state.get_state_array())
    total_time_double_flipping_un_optimized += b - a
    total_time_double_flipping_optimized += c - b
    if np.abs(res_double_flip_a_same[0] - res_double_flip_b_same[0]) > 1e-6:
        print("Difference for Double Flipping Same Site")
        print(res_double_flip_a_same[0])
        print(res_double_flip_b_same[0])

    val_from_manual_caculation = ham_canonical.get_H_eff(
        time=measurement_time, system_state=use_state
    ) - ham_canonical.get_H_eff(
        time=measurement_time, system_state=samesite_double_flipped_copy
    )
    if np.abs(res_double_flip_a_same[0] - val_from_manual_caculation) > 1e-6:
        print("Difference for Double Flipping Same Site with manual")
        print(val_from_manual_caculation)
        print(res_double_flip_a_same[0])

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

    val_from_manual_caculation = ham_canonical.get_H_eff(
        time=measurement_time, system_state=use_state
    ) - ham_canonical.get_H_eff(
        time=measurement_time, system_state=adjacent_double_flipped_copy
    )
    if np.abs(res_double_flip_a[0] - val_from_manual_caculation) > 1e-6:
        print("Difference for Double Flipping with manual")
        print(val_from_manual_caculation)
        print(res_double_flip_a[0])

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
    res_double_flip_b_far = ham_flip_optimized.get_H_eff_difference_double_flipping(
        time=measurement_time,
        flipping1_up=sw1_up,
        flipping1_index=sw1_index,
        flipping2_up=sw3_up,
        flipping2_index=sw3_index,
        before_swap_system_state=use_state,
    )
    c = measure() * 1000
    compare_arrays(modification_check_comparearray, use_state.get_state_array())
    total_time_double_flipping_un_optimized += b - a
    total_time_double_flipping_optimized += c - b
    if np.abs(res_double_flip_a_far[0] - res_double_flip_b_far[0]) > 1e-6:
        print("Difference for far Double Flipping")
        print(res_double_flip_a_far[0])
        print(res_double_flip_b_far[0])

    val_from_manual_caculation = ham_canonical.get_H_eff(
        time=measurement_time, system_state=use_state
    ) - ham_canonical.get_H_eff(
        time=measurement_time, system_state=far_double_flipped_copy
    )
    if np.abs(res_double_flip_a_far[0] - val_from_manual_caculation) > 1e-6:
        print("Difference for far Double Flipping with manual")
        print(val_from_manual_caculation)
        print(res_double_flip_a_far[0])

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

    a = measure() * 1000
    res_double_flip_a = ham_canonical_second_order.get_H_eff_difference_double_flipping(
        time=measurement_time,
        flipping1_up=sw1_up,
        flipping1_index=sw1_index,
        flipping2_up=sw2_up,
        flipping2_index=sw2_index,
        before_swap_system_state=use_state,
    )
    b = measure() * 1000
    res_double_flip_b = ham_second_order_optimized.get_H_eff_difference_double_flipping(
        time=measurement_time,
        flipping1_up=sw1_up,
        flipping1_index=sw1_index,
        flipping2_up=sw2_up,
        flipping2_index=sw2_index,
        before_swap_system_state=use_state,
    )
    c = measure() * 1000
    compare_arrays(modification_check_comparearray, use_state.get_state_array())
    total_time_double_flipping_un_optimized_second_order += b - a
    total_time_double_flipping_optimized_second_order += c - b
    if np.abs(res_double_flip_a[0] - res_double_flip_b[0]) > 1e-6:
        print("Second Order Difference for Double Flipping")
        print(res_double_flip_a[0])
        print(res_double_flip_b[0])

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

print("SECOND ORDER: hopping ms:", total_time_swapping_un_optimized_second_order)
print("SECOND ORDER: hopping optimized ms:", total_time_swapping_optimized_second_order)

print()

print("SECOND ORDER: flipping ms:", total_time_flipping_un_optimized_second_order)
print(
    "SECOND ORDER: flipping optimized ms:", total_time_flipping_optimized_second_order
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
    "Verified the correctness for the timing optimizations for:", iterations, " cases"
)

system_geometry_chain = systemgeometry.LinearChainNonPeriodicState(3)
initial_system_state_chain = state.HomogenousInitialSystemState(system_geometry)
use_state_chain = state.SystemState(system_geometry_chain, initial_system_state)
hailtonian_chain_plus = hamiltonian.HardcoreBosonicHamiltonianSecondOrder(
    U=U,
    E=E,
    J=J,
    phi=phi,
    initial_system_state=initial_system_state_chain,
    system_geometry=system_geometry_chain,
)
hailtonian_chain_minus = hamiltonian.HardcoreBosonicHamiltonianSecondOrder(
    U=U,
    E=-E,
    J=J,
    phi=phi,
    initial_system_state=initial_system_state_chain,
    system_geometry=system_geometry_chain,
)

iterations_symmetry = 100
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

    plus_energy_gap = hailtonian_chain_plus.get_H_eff_difference_flipping(
        time=measurement_time,
        flipping_up=flip_up,
        flipping_index=index_to_flip,
        before_swap_system_state=use_state_chain,
    )[0]
    minus_energy_gap = hailtonian_chain_minus.get_H_eff_difference_flipping(
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

    plus_energy = hailtonian_chain_plus.get_H_eff(
        time=measurement_time,
        system_state=use_state_chain,
    )
    minus_energy = hailtonian_chain_minus.get_H_eff(
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

print(
    f"Verified the correctness for the locational inversion for {iterations_symmetry} iterations"
)
