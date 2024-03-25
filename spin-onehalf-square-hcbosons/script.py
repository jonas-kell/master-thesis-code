# pip3 install matplotlib
# pip3 install numpy

import numpy as np
from randomgenerator import RandomGenerator
import state
import sampler
import hamiltonian
import observables
import measurements
import multiprocessing
import systemgeometry
from typing import List, Literal

if __name__ == "__main__":
    # ! General Hamiltonian properties
    U = 0.4
    E = -0.4
    J = 0.001
    # must NOT be integer-multiples of np.pi/2 or you get division by zero
    phi = np.pi / 100

    # ! Simulation Scope Settings
    start_time: float = 70
    time_step: float = 7
    number_of_time_steps: int = int(1)

    # ! Control behavioral settings here ----------------------------------------------------
    system_geometry_type: Literal["square_np", "chain"] = "square_np"
    initial_system_state_type: Literal["homogenous", "singular"] = "homogenous"
    hamiltonian_type: Literal[
        "canonical", "swap_optimized", "flip_optimized", "both_optimizations"
    ] = "both_optimizations"
    sampling_strategy: Literal["exact", "monte_carlo"] = "monte_carlo"

    # ! Monte Carlo settings
    mc_modification_mode: Literal["flipping", "hopping"] = "hopping"
    num_monte_carlo_samples: int = 40000  # 3x3 system has 262144 states
    num_samples_per_chain: int = 10  # arbitrary at the moment
    mc_pre_therm_strategy: Literal[
        "vacuum",
        "each_random",
        "specified_level",
        "random_level_uniform",
        "random_level_binomial",
    ] = "random_level_binomial"
    mc_pre_therm_fill_level = 0.5

    # ! Randomizer
    randomness_seed = "aok"
    random_generator = RandomGenerator(randomness_seed)

    # ! Geometry of system
    if system_geometry_type == "square_np":  # type: ignore - switch is hard-coded.
        system_geometry = systemgeometry.SquareSystemNonPeriodicState(2)
    if system_geometry_type == "chain":  # type: ignore - switch is hard-coded.
        system_geometry = systemgeometry.LinearChainNonPeriodicState(4)

    # ! Initial System State
    if initial_system_state_type == "homogenous":  # type: ignore - switch is hard-coded.
        initial_system_state = state.HomogenousInitialSystemState(system_geometry)
    if initial_system_state_type == "singular":  # type: ignore - switch is hard-coded.
        initial_system_state = state.SingularDoubleOccupationInitialSystemState(
            0,
            1000.0,  # larger values would cause overflow/underflow
            system_geometry,
        )

    # Hamiltonian
    if hamiltonian_type == "swap_optimized":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianSwappingOptimization(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
    if hamiltonian_type == "flip_optimized":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianFlippingOptimization(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
    if hamiltonian_type == "both_optimizations":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianFlippingAndSwappingOptimization(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
    if hamiltonian_type == "canonical":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonian(U=U, E=E, J=J, phi=phi)

    # ! Observables that are tested for
    current_from = 0
    current_to = 2
    direction_dependent = True
    obs: List[observables.Observable] = [
        observables.DoubleOccupationAtSite(0, system_geometry),
        observables.DoubleOccupationAtSite(1, system_geometry),
        observables.DoubleOccupationAtSite(2, system_geometry),
        observables.DoubleOccupationAtSite(3, system_geometry),
        # observables.DoubleOccupationFraction(),
        observables.SpinCurrent(
            direction_dependent=direction_dependent,
            site_index_from=current_from,
            site_index_to=current_to,
            spin_up=True,
            system_geometry=system_geometry,
            system_hamiltonian=ham,
        ),
        observables.SpinCurrent(
            direction_dependent=direction_dependent,
            site_index_from=current_from,
            site_index_to=current_to,
            spin_up=False,
            system_geometry=system_geometry,
            system_hamiltonian=ham,
        ),
    ]

    # ! Sampling Strategy
    if sampling_strategy == "monte_carlo":  # type: ignore - switch is hard-coded.
        if initial_system_state_type != "homogenous":  # type: ignore - switch is hard-coded.
            # ! These can NOT be monte carlo sampled as it seems.
            # The answers are way over-inflated, as we seem to have a not smooth-enough energy landscape and can "drop" into the one high energy state for way too long once found
            print(
                "Warning: Non-Homogenous system probably cannot be mc-sampled, because not smooth enough"
            )
        # Step-State-Modification
        if mc_modification_mode == "hopping":  # type: ignore - switch is hard-coded.
            allow_hopping_across_spin_direction = True
            state_modification = state.LatticeNeighborHopping(
                allow_hopping_across_spin_direction=allow_hopping_across_spin_direction,
                system_geometry=system_geometry,
            )
        if mc_modification_mode == "flipping":  # type: ignore - switch is hard-coded.
            state_modification = state.RandomFlipping(
                system_geometry=system_geometry,
            )
        if mc_pre_therm_strategy == "vacuum":  # type: ignore - switch is hard-coded.
            pre_therm_strategy = sampler.VacuumStateBeforeThermalization()
        elif mc_pre_therm_strategy == "specified_level":  # type: ignore - switch is hard-coded.
            pre_therm_strategy = (
                sampler.FillRandomlyToSpecifiedFillLevelBeforeThermalization(
                    mc_pre_therm_fill_level
                )
            )
        elif mc_pre_therm_strategy == "random_level_uniform":  # type: ignore - switch is hard-coded.
            pre_therm_strategy = (
                sampler.FillRandomlyToFillLevelPulledFromUniformDistributionBeforeThermalization()
            )
        elif mc_pre_therm_strategy == "random_level_binomial":  # type: ignore - switch is hard-coded.
            pre_therm_strategy = (
                sampler.FillRandomlyToFillLevelPulledFromBinomialDistributionBeforeThermalization()
            )
        elif mc_pre_therm_strategy == "each_random":  # type: ignore - switch is hard-coded.
            pre_therm_strategy = sampler.EachSiteRandomBeforeThermalization()

        # Monte Carlo Sampler
        num_intermediate_mc_steps: int = 2 * (
            2 * system_geometry.get_number_sites_wo_spin_degree()
        )
        # arbitrary increase for thermalization at the moment
        num_thermalization_steps: int = 10 * num_intermediate_mc_steps
        state_sampler = sampler.MonteCarloSampler(
            system_geometry=system_geometry,
            initial_system_state=initial_system_state,
            system_hamiltonian=ham,
            num_intermediate_mc_steps=num_intermediate_mc_steps,
            num_samples=num_monte_carlo_samples,
            num_thermalization_steps=num_thermalization_steps,
            num_samples_per_chain=num_samples_per_chain,
            state_modification=state_modification,
            before_thermalization_initialization=pre_therm_strategy,
        )
    if sampling_strategy == "exact":  # type: ignore - switch is hard-coded.
        state_sampler = sampler.ExactSampler(
            system_geometry=system_geometry,
            initial_system_state=initial_system_state,
        )

    # ! Simulation
    number_workers = multiprocessing.cpu_count()
    measurements.main_measurement_function(
        start_time=start_time,
        time_step=time_step,
        number_of_time_steps=number_of_time_steps,
        hamiltonian=ham,
        observables=obs,
        random_generator=random_generator,
        state_sampler=state_sampler,
        number_workers=number_workers,
        plot=True,
        plot_title=f"obs for phi={phi/(2*np.pi) * 360:.1f}°, U={U:.2f}, E={E:.2f}, J={J:.5f}",
        write_to_file=True,
    )
