import time
import state
import systemgeometry
import observables
import hamiltonian
import sampler
import numpy as np
from randomgenerator import RandomGenerator
from variationalclassicalnetworks import ChainDirectionDependentAllSameFirstOrder

U = 0.9
E = -0.6
J = 0.3
phi = np.pi / 3
measurement_time = 1.7

n = 8

random_generator = RandomGenerator(str(time.time()))

system_geometry = systemgeometry.LinearChainNonPeriodicState(n)

initial_system_state = state.HomogenousInitialSystemState(system_geometry)

vcn_chain = ChainDirectionDependentAllSameFirstOrder(
    system_geometry=system_geometry, J=J
)

exact_sampler = sampler.ExactSampler(
    system_geometry=system_geometry, initial_system_state=initial_system_state
)

ham = hamiltonian.VCNHardCoreBosonicHamiltonianAnalyticalParamsFirstOrder(
    U=U,
    E=E,
    J=J,
    phi=phi,
    initial_system_state=initial_system_state,
    psi_selection=vcn_chain,
    random_generator=random_generator,
    init_sigma=0,
    eta_calculation_sampler=exact_sampler,
    pseudo_inverse_cutoff=1e-10,
    variational_step_fraction_multiplier=1,
    time_step_size=0,
)
ham_first_order = hamiltonian.HardcoreBosonicHamiltonianFlippingAndSwappingOptimization(
    U=U,
    E=E,
    J=J,
    phi=phi,
    initial_system_state=initial_system_state,
)

energy_observable = observables.Energy(ham=ham, geometry=system_geometry)

use_state = state.SystemState(system_geometry, initial_system_state)

number_tests = 1000
for _ in range(number_tests):
    use_state.init_random_filling(random_generator=random_generator)

    # state_array = use_state.get_state_array()
    # print(state_array[0:n])
    # print(state_array[n:])
    # print(vcn_chain.eval_PSIs_on_state(use_state))

    ham.initialize(time=measurement_time)
    energy_obs = (
        energy_observable.get_expectation_value(
            time=measurement_time, system_state=use_state
        )
        * system_geometry.get_number_sites_wo_spin_degree()
    )
    _, energy_eloc_vcn = ham.calculate_O_k_and_E_loc(
        time=measurement_time, system_state=use_state
    )

    if np.abs(energy_eloc_vcn - energy_obs) > 1e-6:
        print(energy_obs)
        print(energy_eloc_vcn)

    h_eff_o1 = ham_first_order.get_H_eff(time=measurement_time, system_state=use_state)
    h_eff_vcn_analytic = ham.get_H_eff(time=measurement_time, system_state=use_state)

    if np.abs(h_eff_o1 - h_eff_vcn_analytic) > 1e-6:
        print(h_eff_o1)
        print(h_eff_vcn_analytic)
