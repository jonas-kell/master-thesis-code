import time
import state
import systemgeometry
import numpy as np
from randomgenerator import RandomGenerator
from variationalclassicalnetworks import ChainDirectionDependentAllSameFirstOrder

U = 0.3
E = -0.5
J = 1
phi = np.pi / 3
measurement_time = 5 * (1 / J)

n = 4

random = RandomGenerator(str(time.time()))

system_geometry = systemgeometry.LinearChainNonPeriodicState(n)

initial_system_state = state.HomogenousInitialSystemState(system_geometry)

vcn_chain = ChainDirectionDependentAllSameFirstOrder(
    system_geometry=system_geometry, J=J
)

use_state = state.SystemState(system_geometry, initial_system_state)

use_state.init_random_filling(random_generator=random)

state_array = use_state.get_state_array()
print(state_array[0:n])
print(state_array[n:])


print(vcn_chain.eval_PSIs_on_state(use_state))
