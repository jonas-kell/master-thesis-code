import time
from time import time as measure
import state
import systemgeometry
import numpy as np
from randomgenerator import RandomGenerator
from variationalclassicalnetworks import ChainDirectionDependentAllSameFirstOrder


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

random = RandomGenerator(str(time.time()))

# Caution: This breaks from linear Chain, n<=3 because then no 3 different comparison indices can be found
system_geometry = systemgeometry.LinearChainNonPeriodicState(10)

initial_system_state = state.HomogenousInitialSystemState(system_geometry)

use_state = state.SystemState(system_geometry, initial_system_state)

vcn_helper = ChainDirectionDependentAllSameFirstOrder(
    system_geometry=system_geometry, J=J
)

total_time_optimized_single = 0
total_time_un_optimized_single = 0
total_time_optimized_double = 0
total_time_un_optimized_double = 0


iterations = 1000
for _ in range(iterations):
    use_state.init_random_filling(random)

    sw1_up: bool = random.randbool()
    sw1_index: int = random.randint(0, use_state.get_number_sites_wo_spin_degree() - 1)

    sw2_up: bool = random.randbool()
    sw2_index: int = random.randint(0, use_state.get_number_sites_wo_spin_degree() - 1)
    while sw2_index in [sw1_index]:
        sw2_index = random.randint(0, use_state.get_number_sites_wo_spin_degree() - 1)

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


print("Single Flipping")
print(f"total time un-optimized flipping: {total_time_un_optimized_single}")
print(f"total time optimized flipping: {total_time_optimized_single}")

print()
print("Double Flipping")
print(f"total time un-optimized flipping: {total_time_un_optimized_double}")
print(f"total time optimized flipping: {total_time_optimized_double}")
