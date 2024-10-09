# pip3 install matplotlib
# pip3 install numpy

from typing import List, Literal, Union, Type, Dict, cast
import multiprocessing
import argparse
import numpy as np
from randomgenerator import RandomGenerator
import state
import sampler
import hamiltonian
import observables
import measurements
import systemgeometry

AcceptableTypes = Union[Type[int], Type[float], Type[str]]


def get_argument(
    arguments: Dict[str, Union[float, int, str]],
    name: str,
    type: AcceptableTypes,
    default: Union[float, int, str],
):
    extracted_value = default
    try:
        extracted_value = type(arguments[name])
        print(f"Using non-default Argument for param {name}: {extracted_value}")
    except ValueError:
        pass
    except TypeError:
        pass

    return extracted_value


if __name__ == "__main__":
    # ? !! Possible overwrite of Parameters, read them in from the command-line call
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")

    # Expected command-line arguments
    parser.add_argument("--U", required=False)
    parser.add_argument("--E", required=False)
    parser.add_argument("--J", required=False)
    parser.add_argument("--phi", required=False)
    parser.add_argument("--n", required=False)
    parser.add_argument("--number_workers", required=False)
    parser.add_argument("--job_array_index", required=False)

    args = vars(parser.parse_args())

    # ? !! Default values for Parameters:

    # ! General Hamiltonian properties
    U = cast(float, get_argument(args, "U", float, 0.7))
    E = cast(float, get_argument(args, "E", float, -0.3))
    J = cast(float, get_argument(args, "J", float, 0.001))

    n = cast(int, get_argument(args, "n", int, 2))

    # must NOT be integer-multiples of np.pi/2 or you get division by zero
    phi = cast(float, get_argument(args, "phi", float, np.pi / 10))

    # ! Simulation Scope Settings
    start_time: float = 0
    time_step: float = 0.125
    number_of_time_steps: int = int(20)

    # ! Hardware Settings
    cpu_core_count = (
        # ! Caution on HPC, this may NOT be the right amount that is available, because we may only use parts of a CPU
        multiprocessing.cpu_count()
    )
    number_workers = cast(
        int,
        get_argument(
            args,
            "number_workers",
            int,
            cpu_core_count,  # assume we want to utilize all cores
        ),
    )
    job_array_index = cast(
        int,
        get_argument(
            args,
            "job_array_index",
            int,
            0,
        ),
    )

    # ! Control behavioral settings here ----------------------------------------------------
    system_geometry_type: Literal["square_np", "chain"] = "square_np"
    initial_system_state_type: Literal["homogenous", "singular"] = "homogenous"
    hamiltonian_type: Literal[
        "canonical",
        "swap_optimized",
        "flip_optimized",
        "both_optimizations",
        "canonical_legacy_care_for_psi",
    ] = "both_optimizations"
    sampling_strategy: Literal["exact", "monte_carlo"] = "exact"

    # ! Monte Carlo settings
    mc_modification_mode: Literal["flipping", "hopping"] = "hopping"
    mc_thermalization_mode: Literal["flipping", "hopping"] = "hopping"
    num_monte_carlo_samples: int = 40000  # 3x3 system has 262144 states
    num_samples_per_chain: int = 20  # arbitrary at the moment
    mc_pre_therm_strategy: Literal[
        "vacuum",
        "each_random",
        "specified_level",
        "random_level_uniform",
        "random_level_binomial",
    ] = "each_random"
    # only relevant for mc_pre_therm_strategy="specified_level"
    mc_pre_therm_specified_fill_level = 0.5

    # !!!!!!! ABOVE THIS, ONE CAN SET SIMULATION PARAMETERS !!!!!!!!!!!
    # !!!!!!! BELOW THIS, THE VALUES GET USED, NO LONGER CHANGE THEM ONLY COMPUTE WITH THEM !!!!!!!!!!!

    # ! Randomizer
    randomness_seed = "very_nice_seed"
    random_generator = RandomGenerator(randomness_seed)

    # ! Geometry of system
    if system_geometry_type == "square_np":  # type: ignore - switch is hard-coded.
        system_geometry = systemgeometry.SquareSystemNonPeriodicState(n)
    if system_geometry_type == "chain":  # type: ignore - switch is hard-coded.
        system_geometry = systemgeometry.LinearChainNonPeriodicState(n)

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
        ham = hamiltonian.HardcoreBosonicHamiltonian(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
    if hamiltonian_type == "canonical_legacy_care_for_psi":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianStraightCalcPsiDiff(
            U=U, E=E, J=J, phi=phi
        )

    # ! Observables that are tested for
    num_of_sites = system_geometry.get_number_sites_wo_spin_degree()
    check_concurrence = True
    check_concurrence_threshold = 1e-2
    obs = []
    center_of_concurrence_index = 0
    for up1, up2 in [(True, True), (True, False), (False, True), (False, False)]:
        obs_generated: List[observables.Observable] = [
            observables.Concurrence(
                site_index_from=center_of_concurrence_index,
                site_index_to=i,
                spin_up_from=up1,
                spin_up_to=up2,
                system_hamiltonian=ham,
                system_geometry=system_geometry,
                perform_checks=check_concurrence,
                check_threshold=check_concurrence_threshold,
            )
            for i in range(num_of_sites)
            if i != center_of_concurrence_index
            # TODO make it possible to obtain entanglement between same site, but different spin (requires checking and rework of double flipping to allow on-site flipping of both spin degrees)
        ]  # Measure the occupation at ALL sites

        obs += obs_generated

    # # current_from = 0
    # # current_to = 2
    # # direction_dependent = True
    # obs_hard_coded: List[observables.Observable] = [
    #     observables.DoubleOccupationFraction(),
    #     # observables.SpinCurrent(
    #     #     direction_dependent=direction_dependent,
    #     #     site_index_from=current_from,
    #     #     site_index_to=current_to,
    #     #     spin_up=True,
    #     #     system_geometry=system_geometry,
    #     #     system_hamiltonian=ham,
    #     # ),
    #     # observables.SpinCurrent(
    #     #     direction_dependent=direction_dependent,
    #     #     site_index_from=current_from,
    #     #     site_index_to=current_to,
    #     #     spin_up=False,
    #     #     system_geometry=system_geometry,
    #     #     system_hamiltonian=ham,
    #     # ),
    # ]

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
        if mc_thermalization_mode == "hopping":  # type: ignore - switch is hard-coded.
            allow_hopping_across_spin_direction = True
            state_modification_thermalization = state.LatticeNeighborHopping(
                allow_hopping_across_spin_direction=allow_hopping_across_spin_direction,
                system_geometry=system_geometry,
            )
        if mc_thermalization_mode == "flipping":  # type: ignore - switch is hard-coded.
            state_modification_thermalization = state.RandomFlipping(
                system_geometry=system_geometry,
            )
        if mc_pre_therm_strategy == "vacuum":  # type: ignore - switch is hard-coded.
            pre_therm_strategy = sampler.VacuumStateBeforeThermalization()
        elif mc_pre_therm_strategy == "specified_level":  # type: ignore - switch is hard-coded.
            pre_therm_strategy = (
                sampler.FillRandomlyToSpecifiedFillLevelBeforeThermalization(
                    mc_pre_therm_specified_fill_level
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
            state_modification_thermalization=state_modification_thermalization,
            before_thermalization_initialization=pre_therm_strategy,
        )
    if sampling_strategy == "exact":  # type: ignore - switch is hard-coded.
        state_sampler = sampler.ExactSampler(
            system_geometry=system_geometry,
            initial_system_state=initial_system_state,
        )

    # ! Simulation
    measurements.main_measurement_function(
        start_time=start_time,
        time_step=time_step,
        number_of_time_steps=number_of_time_steps,
        hamiltonian=ham,
        observables=obs,
        random_generator=random_generator,
        state_sampler=state_sampler,
        number_workers=number_workers,
        job_array_index=job_array_index,
        plot=True,  # do not plot on the HPC-Server!
        plot_title=f"O for phi={phi/(2*np.pi) * 360:.1f}°, U={U:.2f}, E={E:.2f}, J={J:.5f}",
        write_to_file=True,
    )
