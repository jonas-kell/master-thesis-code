import numpy as np
from randomgenerator import RandomGenerator
import state
import sampler
import hamiltonian
import observables
import measurements

if __name__ == "__main__":
    randomness_seed = "k"

    U = 0.4
    E = 0.4
    J = 0.001

    # phi = np.pi # orthogonal to linear chain, so should not change due to electrical field
    # phi = 0  # linear chain should change due to electrical field
    phi = np.pi / 2 - np.pi / 100

    random_generator = RandomGenerator(randomness_seed)
    ham = hamiltonian.HardcoreBosonicHamiltonian(U=U, E=E, J=J, phi=phi)

    # state
    system_state = state.SquareSystemNonPeriodicState(2)
    # system_state = state.LinearChainNonPeriodicState(4)

    # observables
    # obs = [observables.DoubleOccupationFraction()]
    obs = [
        observables.DoubleOccupationAtSite(0, system_state),
        observables.DoubleOccupationAtSite(1, system_state),
        observables.DoubleOccupationAtSite(2, system_state),
        observables.DoubleOccupationAtSite(3, system_state),
        observables.DoubleOccupationFraction(),
    ]

    no_monte_carlo_samples: int = 40000  # 3x3 system has 262144 states
    no_intermediate_mc_steps: int = 2 * (
        2 * system_state.get_number_sites_wo_spin_degree()
    )
    no_thermalization_steps: int = 10 * no_intermediate_mc_steps
    no_random_flips: int = 1
    starting_fill_level: float = 0.5

    state_sampler = sampler.MonteCarloSampler(
        system_state=system_state,
        system_hamiltonian=ham,
        random_generator=random_generator,
        no_intermediate_mc_steps=no_intermediate_mc_steps,
        no_random_flips=no_random_flips,
        no_samples=no_monte_carlo_samples,
        no_thermalization_steps=no_thermalization_steps,
        initial_fill_level=starting_fill_level,
    )

    sample_exactly = False
    sample_exactly = True
    if sample_exactly:
        state_sampler = sampler.ExactSampler(
            system_state=system_state,
        )

    start_time: float = 0
    time_step: float = 0.3
    number_of_time_steps: int = int(13)

    measurements.main_measurement_function(
        start_time=start_time,
        time_step=time_step,
        number_of_time_steps=number_of_time_steps,
        hamiltonian=ham,
        observables=obs,
        state_sampler=state_sampler,
        plot=True,
        plot_title=f"obs for phi = {phi/(2*np.pi) * 360:.1f}°",
    )
