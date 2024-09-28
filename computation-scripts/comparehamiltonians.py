import time
import state
import systemgeometry
import numpy as np
import hamiltonian
from randomgenerator import RandomGenerator
import datetime

U = 0.3
E = -0.4
J = 1
phi = np.pi / 3

random = RandomGenerator(str(time.time()))

system_geometry = systemgeometry.SquareSystemNonPeriodicState(2)

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

for _ in range(10):
    use_state.init_random_filling(random)

    measurement_time = 1

    sw1_up: bool = random.randbool()
    sw1_index: int = 2
    sw2_up: bool = random.randbool()
    sw2_index: int = 1

    a = datetime.datetime.now()
    res_new = ham_canonical.get_H_eff(
        time=measurement_time,
        system_state=use_state,
    )
    b = datetime.datetime.now()
    res_legacy = ham_canonical_legacy.get_H_eff(
        time=measurement_time,
        system_state=use_state,
    )
    c = datetime.datetime.now()
    total_time_new += int((b - a).total_seconds() * 1000)
    total_time_legacy += int((c - b).total_seconds() * 1000)
    if np.abs(res_new - res_legacy) > 1e-6:
        print("Difference for New implementation")
        print(use_state.get_state_array())
        print(res_new)
        print(res_legacy)

    # res_a = ham_canonical.get_H_eff_difference_swapping(
    #     time=measurement_time,
    #     sw1_up=sw1_up,
    #     sw1_index=sw1_index,
    #     sw2_up=sw2_up,
    #     sw2_index=sw2_index,
    #     before_swap_system_state=use_state,
    # )
    # res_b = ham_swap_optimized.get_H_eff_difference_swapping(
    #     time=measurement_time,
    #     sw1_up=sw1_up,
    #     sw1_index=sw1_index,
    #     sw2_up=sw2_up,
    #     sw2_index=sw2_index,
    #     before_swap_system_state=use_state,
    # )
    # if np.abs(res_a[0] - res_b[0]) > 1e-6:
    #     print("Difference for Hopping")
    #     print(res_a[0])
    #     print(res_b[0])

    # res_flip_a = ham_canonical.get_H_eff_difference_flipping(
    #     time=measurement_time,
    #     flipping_up=sw1_up,
    #     flipping_index=sw1_index,
    #     before_swap_system_state=use_state,
    # )
    # res_flip_b = ham_flip_optimized.get_H_eff_difference_flipping(
    #     time=measurement_time,
    #     flipping_up=sw1_up,
    #     flipping_index=sw1_index,
    #     before_swap_system_state=use_state,
    # )
    # if np.abs(res_flip_a[0] - res_flip_b[0]) > 1e-6:
    #     print("Difference for Flipping")
    #     print(res_flip_a[0])
    #     print(res_flip_b[0])

print("new ms:", total_time_new)
print("legacy ms:", total_time_legacy)
