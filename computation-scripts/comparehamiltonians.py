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
phi = np.pi / 3

random = RandomGenerator(str(time.time()))

system_geometry = systemgeometry.SquareSystemNonPeriodicState(5)

initial_system_state = state.HomogenousInitialSystemState(system_geometry)

ham_canonical_legacy = hamiltonian.HardcoreBosonicHamiltonianStraightCalcPsiDiff(
    U=U, E=E, J=J, phi=phi
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

use_state = state.SystemState(system_geometry, initial_system_state)

total_time_new = 0
total_time_legacy = 0

total_time_swapping_un_optimized = 0
total_time_swapping_optimized = 0

total_time_flipping_un_optimized = 0
total_time_flipping_optimized = 0

total_time_double_flipping_un_optimized = 0
total_time_double_flipping_optimized = 0

iterations = 1000
for _ in range(iterations):
    use_state.init_random_filling(random)

    measurement_time = 1.5

    sw1_up: bool = random.randbool()
    sw1_index: int = 5
    sw2_up: bool = random.randbool()
    sw2_index: int = 6
    sw3_up: bool = random.randbool()
    sw3_index: int = 3

    if sw2_index not in system_geometry.get_nearest_neighbor_indices(sw1_index):
        raise Exception("Expect to be neighbor")
    if sw3_index in system_geometry.get_nearest_neighbor_indices(sw1_index):
        raise Exception("Expected not to be a neighbor")

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
    total_time_new += b - a
    total_time_legacy += c - b
    if np.abs(res_new - res_legacy) > 1e-6:
        print("Difference for New implementation")
        print(use_state.get_state_array())
        print(int(res_new))
        print(int(res_legacy))

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
    total_time_flipping_un_optimized += b - a
    total_time_flipping_optimized += c - b
    if np.abs(res_flip_a[0] - res_flip_b[0]) > 1e-6:
        print("Difference for Flipping")
        print(res_flip_a[0])
        print(res_flip_b[0])

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
    total_time_double_flipping_un_optimized += b - a
    total_time_double_flipping_optimized += c - b
    if np.abs(res_double_flip_a[0] - res_double_flip_b[0]) > 1e-6:
        print("Difference for Double Flipping")
        print(res_double_flip_a[0])
        print(res_double_flip_b[0])

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
    total_time_double_flipping_un_optimized += b - a
    total_time_double_flipping_optimized += c - b
    if np.abs(res_double_flip_a_far[0] - res_double_flip_b_far[0]) > 1e-6:
        print("Difference for far Double Flipping")
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

print(
    "Verified the correctness for the timing optimizations for:", iterations, " cases"
)
