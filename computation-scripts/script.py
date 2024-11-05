# pip3 install matplotlib
# pip3 install numpy

from typing import List, Literal, Union, Type, Dict, cast
import multiprocessing
import argparse
import numpy as np
from plot import plot_measurements
from randomgenerator import RandomGenerator
import state
import sampler
import hamiltonian
import observables
import measurements
import systemgeometry
import variationalclassicalnetworks

AcceptableTypes = Union[Type[int], Type[float], Type[str]]


def get_argument(
    arguments: Dict[str, Union[float, int, str]],
    name: str,
    acc_type: AcceptableTypes,
    default: Union[float, int, str],
):
    extracted_value = default
    try:
        if arguments[name] is None:
            raise ValueError("None set Value")
        extracted_value = acc_type(arguments[name])
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
    parser.add_argument("--start_time", required=False)
    parser.add_argument("--target_time_in_one_over_j", required=False)
    parser.add_argument("--number_of_time_steps", required=False)
    parser.add_argument("--file_name_overwrite", required=False)
    parser.add_argument("--do_not_plot", required=False)

    args = vars(parser.parse_args())

    # ? !! Default values for Parameters:

    plot = False  # do not plot on the HPC-Server!
    plot_setting = cast(str, get_argument(args, "do_not_plot", str, "do_plot"))
    if plot_setting == "do_plot":
        plot = True

    # ! General Hamiltonian properties
    U = cast(float, get_argument(args, "U", float, 0.7))
    E = cast(float, get_argument(args, "E", float, -0.3))
    J = cast(float, get_argument(args, "J", float, U / 10))

    n = cast(int, get_argument(args, "n", int, 2))

    # must NOT be integer-multiples of np.pi/2 or you get division by zero
    phi = cast(float, get_argument(args, "phi", float, np.pi / 10))

    # ! Simulation Scope Settings
    start_time: float = cast(float, get_argument(args, "start_time", float, 0))
    target_time_in_one_over_j: float = cast(
        float, get_argument(args, "target_time_in_one_over_j", float, 8)
    )
    if np.abs(J) < 1e-5:
        # if J interaction deactivated, scale with U
        scaler_factor = U
    else:
        scaler_factor = J
    target_time: float = (1 / np.abs(scaler_factor)) * target_time_in_one_over_j
    number_of_time_steps: int = cast(
        int, get_argument(args, "number_of_time_steps", int, 60)
    )
    time_step: float = (target_time - start_time) / number_of_time_steps

    # ! verification settings
    check_observable_imag = False
    check_observable_imag_threshold = 1e-4

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
    file_name_overwrite_proxy = cast(
        str, get_argument(args, "file_name_overwrite", str, "None")
    )
    file_name_overwrite = (
        None if file_name_overwrite_proxy == "None" else file_name_overwrite_proxy
    )

    # ! Control behavioral settings here ----------------------------------------------------
    system_geometry_type: Literal["square_np", "chain"] = "chain"
    initial_system_state_type: Literal["homogenous", "singular"] = "homogenous"
    hamiltonian_type: Literal[
        "exact",
        "canonical",
        "canonical_second_order",
        "swap_optimized",
        "flip_optimized",
        "both_optimizations",
        "canonical_legacy_care_for_psi",
        "both_optimizations_second_order",
        "variational_classical_networks",
    ] = "variational_classical_networks"
    sampling_strategy: Literal["exact", "monte_carlo"] = "exact"

    # ! VCN settings
    init_sigma: float = 0.001
    max_eta_training_rounds: int = 1000
    min_eta_change_for_abort: float = 0.01
    step_size_factor_h: float = 0.01
    psi_selection_type: Literal["chain_canonical"] = "chain_canonical"

    # ! Monte Carlo settings
    mc_modification_mode: Literal["flipping", "hopping"] = "flipping"
    mc_thermalization_mode: Literal["flipping", "hopping"] = "flipping"
    num_monte_carlo_samples: int = 4000  # 3x3 system has 262144 states
    num_samples_per_chain: int = 80  # arbitrary at the moment
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

    # Psi-Selection (in case of)
    if psi_selection_type == "chain_canonical":  # type: ignore - switch is hard-coded.
        psi_selection = (
            variationalclassicalnetworks.ChainDirectionDependentAllSameFirstOrder(
                J=J, system_geometry=system_geometry
            )
        )

    # TODO make this be monte carlo applicable also
    eta_training_sampler = sampler.ExactSampler(
        system_geometry=system_geometry, initial_system_state=initial_system_state
    )

    # Hamiltonian
    if hamiltonian_type == "exact":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianExact(
            U=U,
            E=E,
            J=J,
            phi=phi,
            system_geometry=system_geometry,
            exact_sampler=sampler.ExactSampler(
                system_geometry=system_geometry,
                initial_system_state=initial_system_state,
            ),
            number_of_workers=number_workers,
        )
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
    if hamiltonian_type == "both_optimizations_second_order":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianFlippingAndSwappingOptimizationSecondOrder(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            system_geometry=system_geometry,
        )
    if hamiltonian_type == "canonical":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonian(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
    if hamiltonian_type == "canonical_second_order":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianSecondOrder(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            system_geometry=system_geometry,
        )
    if hamiltonian_type == "canonical_legacy_care_for_psi":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianStraightCalcPsiDiffFirstOrder(
            U=U, E=E, J=J, phi=phi
        )
    if hamiltonian_type == "variational_classical_networks":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.VCNHardCoreBosonicHamiltonian(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            random_generator=random_generator,
            psi_selection=psi_selection,
            init_sigma=init_sigma,
            eta_training_sampler=eta_training_sampler,
            max_eta_training_rounds=max_eta_training_rounds,
            min_eta_change_for_abort=min_eta_change_for_abort,
            step_size_factor_h=step_size_factor_h,
        )

    # ! Observables that are tested for
    num_of_sites = system_geometry.get_number_sites_wo_spin_degree()
    obs = []

    current_from = 0
    current_to = 1
    direction_dependent = True
    obs_hard_coded: List[observables.Observable] = [
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
        observables.DoubleOccupationAtSite(
            site=current_from,
            system_geometry=system_geometry,
        ),
        observables.DoubleOccupationAtSite(
            site=current_to,
            system_geometry=system_geometry,
        ),
        observables.OccupationAtSite(
            site=current_from,
            up=True,
            system_geometry=system_geometry,
        ),
        observables.OccupationAtSite(
            site=current_from,
            up=False,
            system_geometry=system_geometry,
        ),
        observables.OccupationAtSite(
            site=current_to,
            up=True,
            system_geometry=system_geometry,
        ),
        observables.OccupationAtSite(
            site=current_to,
            up=False,
            system_geometry=system_geometry,
        ),
        observables.Purity(
            site_index_from=current_from,
            site_index_to=current_to,
            spin_up_from=True,
            spin_up_to=True,
            system_hamiltonian=ham,
            system_geometry=system_geometry,
            perform_checks=check_observable_imag,
            check_threshold=check_observable_imag_threshold,
        ),
        observables.Concurrence(
            site_index_from=current_from,
            site_index_to=current_to,
            spin_up_from=True,
            spin_up_to=True,
            system_hamiltonian=ham,
            system_geometry=system_geometry,
            perform_checks=check_observable_imag,
            check_threshold=check_observable_imag_threshold,
        ),
        observables.ConcurrenceAsymm(
            site_index_from=current_from,
            site_index_to=current_to,
            spin_up_from=True,
            spin_up_to=True,
            system_hamiltonian=ham,
            system_geometry=system_geometry,
            perform_checks=check_observable_imag,
            check_threshold=check_observable_imag_threshold,
        ),
    ]
    obs += obs_hard_coded

    for i in range(16):
        obs_generated: List[observables.Observable] = [
            observables.PauliMeasurement(
                site_index_from=current_from,
                site_index_to=current_to,
                spin_up_from=True,
                spin_up_to=True,
                system_hamiltonian=ham,
                system_geometry=system_geometry,
                perform_checks=check_observable_imag,
                check_threshold=check_observable_imag_threshold,
                index_of_pauli_op=i,
            )
        ]

        obs += obs_generated

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
    (time_list, values_list) = measurements.main_measurement_function(
        start_time=start_time,
        time_step=time_step,
        number_of_time_steps=number_of_time_steps,
        hamiltonian=ham,
        observables=obs,
        random_generator=random_generator,
        state_sampler=state_sampler,
        number_workers=number_workers,
        job_array_index=job_array_index,
        write_to_file=True,
        file_name_overwrite=file_name_overwrite,
        check_obs_imag=check_observable_imag,
        check_obs_imag_threshold=check_observable_imag_threshold,
    )

    # ! Plotting
    plot_title = (
        f"O for phi={phi/(2*np.pi) * 360:.1f}°, U={U:.2f}, E={E:.2f}, J={J:.5f}",
    )
    plot_x_label: str = "time t"

    if plot:
        plot_measurements(
            times=time_list,
            values=values_list,
            observable_labels=[observable.get_label() for observable in obs],
            title=plot_title,
            x_label=plot_x_label,
            params=(
                U,
                E,
                J,
            ),
            time_unit_type="one_over_J",
        )
