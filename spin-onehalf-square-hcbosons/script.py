import numpy as np
from randomgenerator import RandomGenerator
import state
import sampler
import hamiltonian
import observables
import measurements
import multiprocessing
import systemgeometry
from typing import List

if __name__ == "__main__":
    randomness_seed = "ok"

    # ! General Hamiltonian properties
    U = 0.4
    E = -0.4
    J = 0.001
    # must not be integer-multiples of np.pi/2 or you get division by zero
    phi = np.pi / 100

    random_generator = RandomGenerator(randomness_seed)
    ham = hamiltonian.HardcoreBosonicHamiltonian(U=U, E=E, J=J, phi=phi)

    # ! Geometry of system
    # system_geometry = systemgeometry.LinearChainNonPeriodicState(4)
    system_geometry = systemgeometry.SquareSystemNonPeriodicState(2)

    # ! Initial System State
    initial_system_state = state.SingularDoubleOccupationInitialSystemState(
        0, 1000.0, system_geometry
    )
    # initial_system_state = state.HomogenousInitialSystemState(system_state)

    # ! Observables that are tested for
    # obs: List[observables.Observable] = [observables.DoubleOccupationFraction()]
    obs: List[observables.Observable] = [
        observables.DoubleOccupationAtSite(0, system_geometry),
        observables.DoubleOccupationAtSite(1, system_geometry),
        observables.DoubleOccupationAtSite(2, system_geometry),
        observables.DoubleOccupationAtSite(3, system_geometry),
        observables.DoubleOccupationFraction(),
    ]

    # ! Sampling Strategy
    # Monte Carlo Sampler
    num_monte_carlo_samples: int = 20000  # 3x3 system has 262144 states
    num_intermediate_mc_steps: int = 2 * (
        2 * system_geometry.get_number_sites_wo_spin_degree()
    )
    num_thermalization_steps: int = 10 * num_intermediate_mc_steps
    num_samples_per_chain: int = int(300)  # arbitrary at the moment
    num_random_flips: int = 1
    starting_fill_level: float = 0.5
    state_sampler = sampler.MonteCarloSampler(
        system_geometry=system_geometry,
        initial_system_state=initial_system_state,
        system_hamiltonian=ham,
        random_generator=random_generator,
        num_intermediate_mc_steps=num_intermediate_mc_steps,
        num_random_flips=num_random_flips,
        num_samples=num_monte_carlo_samples,
        num_thermalization_steps=num_thermalization_steps,
        initial_fill_level=starting_fill_level,
        num_samples_per_chain=num_samples_per_chain,
    )
    # Exact Sampler
    sample_exactly = False
    sample_exactly = True
    if sample_exactly:
        state_sampler = sampler.ExactSampler(
            system_geometry=system_geometry,
            initial_system_state=initial_system_state,
        )

    # ! Simulation Scope settings
    number_workers = multiprocessing.cpu_count()

    start_time: float = 0
    time_step: float = 0.5
    number_of_time_steps: int = int(15)

    measurements.main_measurement_function(
        start_time=start_time,
        time_step=time_step,
        number_of_time_steps=number_of_time_steps,
        hamiltonian=ham,
        observables=obs,
        state_sampler=state_sampler,
        number_workers=number_workers,
        plot=True,
        plot_title=f"obs for phi={phi/(2*np.pi) * 360:.1f}°, U={U:.2f}, E={E:.2f}, J={J:.5f}",
        write_to_file=True,
    )
