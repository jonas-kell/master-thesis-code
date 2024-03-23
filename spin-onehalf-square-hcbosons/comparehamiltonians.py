# import warnings
# import numpy as np
# import debugpy
# warnings.filterwarnings("error")
# try:
#     test = np.complex128(1)
#     for i in range(1000):
#         test *= 10
# except RuntimeWarning:
#     debugpy.breakpoint()

import state
import systemgeometry
import numpy as np
import hamiltonian
from randomgenerator import RandomGenerator
import time

U = 0.3
E = -0.4
J = 1
phi = np.pi / 3

random = RandomGenerator(str(time.time()))

system_geometry = systemgeometry.SquareSystemNonPeriodicState(6)

initial_system_state = state.HomogenousInitialSystemState(system_geometry)

ham_canonical = hamiltonian.HardcoreBosonicHamiltonian(
    U=U,
    E=E,
    J=J,
    phi=phi,
)
ham_swap_optimized = hamiltonian.HardcoreBosonicHamiltonianSwappingOptimization(
    U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
)

use_state = state.SystemState(system_geometry, initial_system_state)

for _ in range(12):
    use_state.init_random_filling(random)
    print(use_state.get_state_array())

    measurement_time = 1.5

    sw1_up: bool = random.randbool()
    sw1_index: int = 2
    sw2_up: bool = random.randbool()
    sw2_index: int = 1

    res_a = ham_canonical.get_H_eff_difference_swapping(
        time=measurement_time,
        sw1_up=sw1_up,
        sw1_index=sw1_index,
        sw2_up=sw2_up,
        sw2_index=sw2_index,
        before_swap_system_state=use_state,
    )
    res_b = ham_swap_optimized.get_H_eff_difference_swapping(
        time=measurement_time,
        sw1_up=sw1_up,
        sw1_index=sw1_index,
        sw2_up=sw2_up,
        sw2_index=sw2_index,
        before_swap_system_state=use_state,
    )
    if np.abs(res_a[0] - res_b[0]) > 1e-6:
        print(res_a[0])
        print(res_b[0])
