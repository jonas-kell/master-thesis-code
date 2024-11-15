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
    parser.add_argument(
        "--record_hamiltonian_properties", action="store_true", required=False
    )
    parser.add_argument("--U", required=False)
    parser.add_argument("--E", required=False)
    parser.add_argument("--J", required=False)
    parser.add_argument("--phi", required=False)
    parser.add_argument("--n", required=False)
    parser.add_argument("--number_workers", required=False)
    parser.add_argument("--job_array_index", required=False)
    parser.add_argument("--start_time", required=False)
    parser.add_argument("--target_time_in_one_over_j", required=False)
    parser.add_argument("--target_time_in_one_over_u", required=False)
    parser.add_argument("--number_of_time_steps", required=False)
    parser.add_argument("--file_name_overwrite", required=False)
    parser.add_argument("--do_not_plot", required=False)
    parser.add_argument("--hamiltonian_type", required=False)
    parser.add_argument("--sampling_strategy", required=False)
    parser.add_argument("--randomnes_seed", required=False)
    parser.add_argument("--num_samples_per_chain", required=False)
    parser.add_argument("--num_monte_carlo_samples", required=False)
    parser.add_argument("--mc_thermalization_mode", required=False)
    parser.add_argument("--mc_modification_mode", required=False)
    parser.add_argument("--system_geometry_type", required=False)
    parser.add_argument("--variational_step_fraction_multiplier", required=False)
    parser.add_argument("--init_sigma", required=False)

    args = vars(parser.parse_args())

    time_default = -123123123

    # ? !! Default values for Parameters:

    plot = False  # do not plot on the HPC-Server!
    plot_setting = cast(str, get_argument(args, "do_not_plot", str, "do_not_plot"))
    if plot_setting == "do_plot":
        plot = True

    # ! General Hamiltonian properties
    U = cast(float, get_argument(args, "U", float, 1.0))
    E = cast(float, get_argument(args, "E", float, 2.5))
    J = cast(float, get_argument(args, "J", float, U / 10))

    n = cast(int, get_argument(args, "n", int, 2))

    # must NOT be integer-multiples of np.pi/2 or you get division by zero
    phi = cast(float, get_argument(args, "phi", float, np.pi / 10))

    # ! Simulation Scope Settings
    start_time: float = cast(float, get_argument(args, "start_time", float, 0))
    target_time_in_one_over_j: float = cast(
        float, get_argument(args, "target_time_in_one_over_j", float, time_default)
    )
    target_time_in_one_over_u: float = cast(
        float, get_argument(args, "target_time_in_one_over_u", float, time_default)
    )
    if (
        target_time_in_one_over_j != time_default
        and target_time_in_one_over_u != time_default
    ):
        raise Exception("Should not have two different target times set")

    target_time_in_one_over_scaler = 5
    scaler_factor = U
    if target_time_in_one_over_j != time_default:
        target_time_in_one_over_scaler = target_time_in_one_over_j
        scaler_factor = J
    if target_time_in_one_over_u != time_default:
        target_time_in_one_over_scaler = target_time_in_one_over_u
        scaler_factor = U

    target_time: float = (1 / np.abs(scaler_factor)) * target_time_in_one_over_scaler
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
    # singular basically only for testing in the beginning
    initial_system_state_type: Literal["homogenous", "singular"] = "homogenous"
    system_geometry_type: Literal["square_np", "chain"] = cast(
        str, get_argument(args, "system_geometry_type", str, "chain")
    )
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
        "variational_classical_networks_analytical_factors",
        "base_energy_only",
    ] = cast(str, get_argument(args, "hamiltonian_type", str, "exact"))
    sampling_strategy: Literal["exact", "monte_carlo"] = cast(
        str, get_argument(args, "sampling_strategy", str, "exact")
    )

    # ! Monte Carlo settings
    mc_modification_mode: Literal["flipping", "hopping"] = cast(
        str, get_argument(args, "mc_modification_mode", str, "flipping")
    )
    mc_thermalization_mode: Literal["flipping", "hopping"] = cast(
        str, get_argument(args, "mc_thermalization_mode", str, "flipping")
    )
    num_monte_carlo_samples: int = cast(
        int,
        get_argument(
            args,
            "num_monte_carlo_samples",
            int,
            4000,  # 3x3 system has 262144 states
        ),
    )
    num_samples_per_chain: int = cast(
        int,
        get_argument(
            args,
            "num_samples_per_chain",
            int,
            80,
        ),
    )
    mc_pre_therm_strategy: Literal[
        "vacuum",
        "each_random",
        "specified_level",
        "random_level_uniform",
        "random_level_binomial",
    ] = "each_random"
    # only relevant for mc_pre_therm_strategy="specified_level"
    mc_pre_therm_specified_fill_level = 0.5
    randomnes_seed = cast(
        str, get_argument(args, "randomnes_seed", str, "very_nice_seed")
    )

    # ! VCN settings
    record_hamiltonian_properties: bool = args["record_hamiltonian_properties"]
    init_sigma: float = cast(float, get_argument(args, "init_sigma", float, 0.001))
    pseudo_inverse_cutoff: float = 1e-10
    psi_selection_type: Literal["chain_canonical"] = "chain_canonical"
    variational_step_fraction_multiplier: int = cast(
        int, get_argument(args, "variational_step_fraction_multiplier", int, 1)
    )

    # !!!!!!! ABOVE THIS, ONE CAN SET SIMULATION PARAMETERS (if not overwritten by input arguments) !!!!!!!!!!!
    # !!!!!!! BELOW THIS, THE VALUES GET USED, NO LONGER CHANGE THEM ONLY COMPUTE WITH THEM !!!!!!!!!!!

    # ! Randomizer
    random_generator = RandomGenerator(randomnes_seed)

    # ! Geometry of system
    if system_geometry_type == "square_np":  # type: ignore - switch is hard-coded.
        system_geometry = systemgeometry.SquareSystemNonPeriodicState(n)
    elif system_geometry_type == "chain":  # type: ignore - switch is hard-coded.
        system_geometry = systemgeometry.LinearChainNonPeriodicState(n)
    else:
        raise Exception("Invalid arguments")

    # ! Initial System State
    if initial_system_state_type == "homogenous":  # type: ignore - switch is hard-coded.
        initial_system_state = state.HomogenousInitialSystemState(system_geometry)
    elif initial_system_state_type == "singular":  # type: ignore - switch is hard-coded.
        initial_system_state = state.SingularDoubleOccupationInitialSystemState(
            0,
            1000.0,  # larger values would cause overflow/underflow
            system_geometry,
        )
    else:
        raise Exception("Invalid arguments")

    # Psi-Selection (in case of)
    if psi_selection_type == "chain_canonical":  # type: ignore - switch is hard-coded.
        psi_selection = (
            variationalclassicalnetworks.ChainDirectionDependentAllSameFirstOrder(
                J=J, system_geometry=system_geometry
            )
        )
    else:
        raise Exception("Invalid arguments")

    # TODO make this be monte carlo applicable also
    eta_calculation_sampler = sampler.ExactSampler(
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
    elif hamiltonian_type == "swap_optimized":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianSwappingOptimization(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
    elif hamiltonian_type == "flip_optimized":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianFlippingOptimization(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
    elif hamiltonian_type == "both_optimizations":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianFlippingAndSwappingOptimization(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
    elif hamiltonian_type == "both_optimizations_second_order":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianFlippingAndSwappingOptimizationSecondOrder(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            system_geometry=system_geometry,
        )
    elif hamiltonian_type == "base_energy_only":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.ZerothOrderFlippingAndSwappingOptimization(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
        )
    elif hamiltonian_type == "canonical":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonian(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
    elif hamiltonian_type == "canonical_second_order":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianSecondOrder(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            system_geometry=system_geometry,
        )
    elif hamiltonian_type == "canonical_legacy_care_for_psi":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.HardcoreBosonicHamiltonianStraightCalcPsiDiffFirstOrder(
            U=U, E=E, J=J, phi=phi
        )
    elif hamiltonian_type == "variational_classical_networks":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.VCNHardCoreBosonicHamiltonian(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            random_generator=random_generator,
            psi_selection=psi_selection,
            init_sigma=init_sigma,
            eta_calculation_sampler=eta_calculation_sampler,
            pseudo_inverse_cutoff=pseudo_inverse_cutoff,
            variational_step_fraction_multiplier=variational_step_fraction_multiplier,
        )
    elif hamiltonian_type == "variational_classical_networks_analytical_factors":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.VCNHardCoreBosonicHamiltonianAnalyticalParamsFirstOrder(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            random_generator=random_generator,
            psi_selection=psi_selection,
            init_sigma=init_sigma,
            eta_calculation_sampler=eta_calculation_sampler,
            pseudo_inverse_cutoff=pseudo_inverse_cutoff,
            variational_step_fraction_multiplier=variational_step_fraction_multiplier,
        )
    else:
        raise Exception("Invalid arguments")

    # ! Observables that are tested for
    num_of_sites = system_geometry.get_number_sites_wo_spin_degree()
    obs = []

    current_from_left = 0
    current_to_left = current_from_left + 1
    current_from_center = n // 2
    current_to_center = current_from_center + 1

    direction_dependent = True
    obs_hard_coded: List[observables.Observable] = [
        observables.SpinCurrent(
            direction_dependent=direction_dependent,
            site_index_from=current_from_left,
            site_index_to=current_to_left,
            spin_up=True,
            system_geometry=system_geometry,
            system_hamiltonian=ham,
        ),
        observables.SpinCurrent(
            direction_dependent=direction_dependent,
            site_index_from=current_from_center,
            site_index_to=current_to_center,
            spin_up=True,
            system_geometry=system_geometry,
            system_hamiltonian=ham,
        ),
        observables.DoubleOccupationAtSite(
            site=current_from_left,
            system_geometry=system_geometry,
        ),
        observables.DoubleOccupationAtSite(
            site=current_to_left,
            system_geometry=system_geometry,
        ),
        observables.DoubleOccupationAtSite(
            site=current_from_center,
            system_geometry=system_geometry,
        ),
        observables.DoubleOccupationAtSite(
            site=current_to_center,
            system_geometry=system_geometry,
        ),
        observables.OccupationAtSite(
            site=current_from_left,
            up=True,
            system_geometry=system_geometry,
        ),
        observables.OccupationAtSite(
            site=current_to_left,
            up=True,
            system_geometry=system_geometry,
        ),
        observables.OccupationAtSite(
            site=current_from_center,
            up=True,
            system_geometry=system_geometry,
        ),
        observables.OccupationAtSite(
            site=current_to_center,
            up=True,
            system_geometry=system_geometry,
        ),
    ]
    obs += obs_hard_coded

    if record_hamiltonian_properties:
        obs_ham_properties: List[observables.Observable] = []
        num_etas = 6  # TODO lift this hard-coded number of constants
        for real_part_of_property in [True, False]:
            for eta_index in range(num_etas):
                obs_ham_properties.append(
                    observables.VCNFactor(
                        ham=ham,
                        param_index=eta_index,
                        param_real_part=real_part_of_property,
                    )
                )

        obs += obs_ham_properties

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
        elif mc_modification_mode == "flipping":  # type: ignore - switch is hard-coded.
            state_modification = state.RandomFlipping(
                system_geometry=system_geometry,
            )
        else:
            raise Exception("Invalid arguments")

        if mc_thermalization_mode == "hopping":  # type: ignore - switch is hard-coded.
            allow_hopping_across_spin_direction = True
            state_modification_thermalization = state.LatticeNeighborHopping(
                allow_hopping_across_spin_direction=allow_hopping_across_spin_direction,
                system_geometry=system_geometry,
            )
        elif mc_thermalization_mode == "flipping":  # type: ignore - switch is hard-coded.
            state_modification_thermalization = state.RandomFlipping(
                system_geometry=system_geometry,
            )
        else:
            raise Exception("Invalid arguments")

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
        else:
            raise Exception("Invalid arguments")

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
    elif sampling_strategy == "exact":  # type: ignore - switch is hard-coded.
        state_sampler = sampler.ExactSampler(
            system_geometry=system_geometry,
            initial_system_state=initial_system_state,
        )
    else:
        raise Exception("Invalid arguments")

    # ! Simulation
    (time_list, values_list, plotting_observables) = (
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
            write_to_file=True,
            file_name_overwrite=file_name_overwrite,
            check_obs_imag=check_observable_imag,
            check_obs_imag_threshold=check_observable_imag_threshold,
        )
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
            observable_labels=[
                observable.get_label() for observable in plotting_observables
            ],
            title=plot_title,
            x_label=plot_x_label,
            params=(
                U,
                E,
                J,
            ),
            time_unit_type="one_over_scaler",
        )
