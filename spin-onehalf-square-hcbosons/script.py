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
    J = 0.01
    phi = np.pi / 8  # putting this to /8 completely changes the graph

    random_generator = RandomGenerator(randomness_seed)
    system_state = state.SquareSystemNonPeriodicState(2)
    ham = hamiltonian.HardcoreBosonicHamiltonian(U=U, E=E, J=J, phi=phi)
    obs = observables.DoubleOccupation()

    beta = 0.05
    no_monte_carlo_samples: int = 2000  # 3x3 system has 262144 states
    no_thermalization_steps: int = 1000
    no_intermediate_mc_steps: int = 20
    no_random_flips: int = 1
    starting_fill_level: float = 1.0

    state_sampler = sampler.MonteCarloSampler(
        system_state=system_state,
        beta=beta,
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

    start_time: float = 0.0
    time_step: float = 0.05
    number_of_time_steps: int = int(80 * 20)

    (sampled_times, sampled_values) = measurements.main_measurement_function(
        start_time=start_time,
        time_step=time_step,
        number_of_time_steps=number_of_time_steps,
        hamiltonian=ham,
        observable=obs,
        state_sampler=state_sampler,
    )

    measurements.plot_measurements(
        times=sampled_times,
        values=sampled_values,
        title="Calculations on Spin System",
        x_label="time t",
        y_label="observed value",
    )
