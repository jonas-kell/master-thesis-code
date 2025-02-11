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
    parser.add_argument("--record_imag_part", action="store_true", required=False)
    parser.add_argument("--ue_might_change", action="store_true", required=False)
    parser.add_argument("--ue_variational", action="store_true", required=False)
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
    parser.add_argument("--randomness_seed", required=False)
    parser.add_argument("--num_samples_per_chain", required=False)
    parser.add_argument("--num_monte_carlo_samples", required=False)
    parser.add_argument("--mc_thermalization_mode", required=False)
    parser.add_argument("--mc_modification_mode", required=False)
    parser.add_argument("--system_geometry_type", required=False)
    parser.add_argument("--variational_step_fraction_multiplier", required=False)
    parser.add_argument("--init_sigma", required=False)
    parser.add_argument("--pseudo_inverse_cutoff", required=False)
    parser.add_argument("--observable_set", required=False)
    parser.add_argument("--psi_selection_type", required=False)
    parser.add_argument("--vcn_parameter_init_distribution", required=False)

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
    system_geometry_type: Literal["square", "chain"] = cast(
        str, get_argument(args, "system_geometry_type", str, "chain")
    )
    hamiltonian_type: Literal[
        "exact",
        #
        "first_order_legacy_care_for_psi",
        "first_order_cononical",
        "second_order_canonical",
        "first_order_swap_optimized",
        "first_order_flip_optimized",
        #
        "zeroth_order_optimized",
        "first_order_optimized",
        "second_order_optimized",
        #
        "variational_classical_networks",
        "variational_classical_networks_analytical_factors",
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
    randomness_seed = cast(
        str, get_argument(args, "randomness_seed", str, "very_nice_seed")
    )

    # ! VCN settings
    record_hamiltonian_properties: bool = args["record_hamiltonian_properties"]
    ue_might_change: bool = args["ue_might_change"]
    ue_variational: bool = args["ue_variational"]
    record_imag_part: bool = args["record_imag_part"]
    init_sigma: float = cast(float, get_argument(args, "init_sigma", float, 0.001))
    pseudo_inverse_cutoff: float = cast(
        float, get_argument(args, "pseudo_inverse_cutoff", float, 1e-10)
    )
    psi_selection_type: Literal[
        "chain_canonical", "chain_combined", "square_canonical"
    ] = cast(str, get_argument(args, "psi_selection_type", str, "chain_canonical"))
    variational_step_fraction_multiplier: int = cast(
        int, get_argument(args, "variational_step_fraction_multiplier", int, 1)
    )

    observable_set: Literal[
        "current_and_occupation",
        "concurrence_and_pauli",
        "energy_and_variance",
        "comparison_validation",
        "energy_and_variance_and_entanglement_test",
    ] = cast(str, get_argument(args, "observable_set", str, "current_and_occupation"))

    vcn_parameter_init_distribution: Literal[
        "normal",
        "uniform",
    ] = cast(str, get_argument(args, "vcn_parameter_init_distribution", str, "normal"))
    if (
        vcn_parameter_init_distribution != "normal"
        and vcn_parameter_init_distribution != "uniform"
    ):
        raise Exception("Not supported Distribution")

    # !!!!!!! ABOVE THIS, ONE CAN SET SIMULATION PARAMETERS (if not overwritten by input arguments) !!!!!!!!!!!
    # !!!!!!! BELOW THIS, THE VALUES GET USED, NO LONGER CHANGE THEM ONLY COMPUTE WITH THEM !!!!!!!!!!!

    # ! Randomizer
    random_generator = RandomGenerator(randomness_seed)

    # ! Geometry of system
    if system_geometry_type == "square":  # type: ignore - switch is hard-coded.
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

    if psi_selection_type == "chain_canonical":  # type: ignore - switch is hard-coded.
        psi_selection = (
            variationalclassicalnetworks.ChainDirectionDependentAllSameFirstOrder(
                J=J, system_geometry=system_geometry
            )
        )
    elif psi_selection_type == "chain_combined":  # type: ignore - switch is hard-coded.
        psi_selection = (
            variationalclassicalnetworks.ChainNotDirectionDependentAllSameFirstOrder(
                J=J, system_geometry=system_geometry
            )
        )
    elif psi_selection_type == "square_canonical":  # type: ignore - switch is hard-coded.
        psi_selection = (
            variationalclassicalnetworks.SquareDirectionDependentAllSameFirstOrder(
                J=J, system_geometry=system_geometry
            )
        )
    else:
        raise Exception("Invalid arguments")

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

    # ! Hamiltonian
    if hamiltonian_type == "exact":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.ExactHamiltonian(
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
    elif hamiltonian_type == "first_order_swap_optimized":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.FirstOrderSwappingOptimizedHamiltonian(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
    elif hamiltonian_type == "first_order_flip_optimized":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.FirstOrderFlippingOptimizedHamiltonian(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
    elif hamiltonian_type == "first_order_optimized":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.FirstOrderOptimizedHamiltonian(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
    elif hamiltonian_type == "second_order_optimized":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.SecondOrderOptimizedHamiltonian(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            system_geometry=system_geometry,
        )
    elif hamiltonian_type == "zeroth_order_optimized":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.ZerothOrderOptimizedHamiltonian(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
        )
    elif hamiltonian_type == "first_order_cononical":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.FirstOrderCanonicalHamiltonian(
            U=U, E=E, J=J, phi=phi, initial_system_state=initial_system_state
        )
    elif hamiltonian_type == "second_order_canonical":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.SecondOrderCanonicalHamiltonian(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            system_geometry=system_geometry,
        )
    elif hamiltonian_type == "first_order_legacy_care_for_psi":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.FirstOrderDifferentiatesPsiHamiltonian(U=U, E=E, J=J, phi=phi)
    elif hamiltonian_type == "variational_classical_networks":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.FirstOrderVariationalClassicalNetworkHamiltonian(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            random_generator=random_generator,
            psi_selection=psi_selection,
            init_sigma=init_sigma,
            eta_calculation_sampler=state_sampler,
            pseudo_inverse_cutoff=pseudo_inverse_cutoff,
            variational_step_fraction_multiplier=variational_step_fraction_multiplier,
            time_step_size=time_step,
            number_workers=number_workers,
            ue_might_change=ue_might_change,
            ue_variational=ue_variational,
            vcn_parameter_init_distribution=vcn_parameter_init_distribution,
        )
    elif hamiltonian_type == "variational_classical_networks_analytical_factors":  # type: ignore - switch is hard-coded.
        ham = hamiltonian.FirstOrderVariationalClassicalNetworkAnalyticalParamsHamiltonian(
            U=U,
            E=E,
            J=J,
            phi=phi,
            initial_system_state=initial_system_state,
            random_generator=random_generator,
            psi_selection=psi_selection,
            vcn_parameter_init_distribution=vcn_parameter_init_distribution,
        )
    else:
        raise Exception("Invalid arguments")

    # ! Need to cross-share the hamiltonian reference and sampler reference
    if isinstance(state_sampler, sampler.MonteCarloSampler):
        state_sampler.init_hamiltonian(system_hamiltonian=ham)

    # ! Observables that are tested for
    num_of_sites = system_geometry.get_number_sites_wo_spin_degree()
    obs = []

    if observable_set == "current_and_occupation":
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
    elif observable_set == "concurrence_and_pauli":
        matrix_index_a = 0
        matrix_index_b = 1

        obs_hard_coded: List[observables.Observable] = [
            observables.Concurrence(
                site_index_from=matrix_index_a,
                site_index_to=matrix_index_b,
                spin_up_from=True,
                spin_up_to=True,
                system_hamiltonian=ham,
                system_geometry=system_geometry,
                perform_checks=check_observable_imag,
                check_threshold=check_observable_imag_threshold,
            ),
            observables.ConcurrenceAsymm(
                site_index_from=matrix_index_a,
                site_index_to=matrix_index_b,
                spin_up_from=True,
                spin_up_to=True,
                system_hamiltonian=ham,
                system_geometry=system_geometry,
                perform_checks=check_observable_imag,
                check_threshold=check_observable_imag_threshold,
            ),
            observables.Purity(
                site_index_from=matrix_index_a,
                site_index_to=matrix_index_b,
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
                    site_index_from=matrix_index_a,
                    site_index_to=matrix_index_b,
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
    elif observable_set == "energy_and_variance":
        obs_hard_coded: List[observables.Observable] = [
            observables.Energy(
                ham=ham,
                geometry=system_geometry,
            ),
            observables.EnergyVariance(
                ham=ham,
                geometry=system_geometry,
            ),
        ]
        obs += obs_hard_coded
    elif observable_set == "comparison_validation":
        obs_hard_coded: List[observables.Observable] = [
            observables.Energy(
                ham=ham,
                geometry=system_geometry,
            ),
            observables.EnergyVariance(
                ham=ham,
                geometry=system_geometry,
            ),
            observables.NormalizationComparison(),
            observables.LocalKinEnergyEquivalent(
                spin_up=True,
                system_geometry=system_geometry,
                system_hamiltonian=ham,
            ),
            observables.LocalKinEnergyEquivalent(
                spin_up=False,
                system_geometry=system_geometry,
                system_hamiltonian=ham,
            ),
        ]
        obs += obs_hard_coded
    elif observable_set == "energy_and_variance_and_entanglement_test":
        if not isinstance(system_geometry, systemgeometry.SquareSystemNonPeriodicState):
            raise Exception(
                "Index calculation for this observable only supports square geometry"
            )

        square_sidelengh = system_geometry.size  # is the sidelenght

        if square_sidelengh % 2 != 0:
            raise Exception(
                "Index calculation for this observable only supports even side-lenghts"
            )

        # we search for sidelength 2 the xxx and for 4 the yyy and so on
        #
        #   0  x  0  0
        #   x  x  y  0
        #   0  y  y  0
        #   0  0  0  0
        #

        use_edge_index = square_sidelengh // 2
        tr_index = (use_edge_index - 1) * square_sidelengh + use_edge_index
        bl_index = use_edge_index * square_sidelengh + (use_edge_index - 1)
        br_index = use_edge_index * (square_sidelengh + 1)

        obs_hard_coded: List[observables.Observable] = [
            observables.Energy(
                ham=ham,
                geometry=system_geometry,
            ),
            observables.EnergyVariance(
                ham=ham,
                geometry=system_geometry,
            ),
            observables.SpinCurrent(
                direction_dependent=True,
                site_index_from=tr_index,
                site_index_to=br_index,
                spin_up=True,
                system_geometry=system_geometry,
                system_hamiltonian=ham,
            ),
            observables.SpinCurrent(
                direction_dependent=True,
                site_index_from=bl_index,
                site_index_to=br_index,
                spin_up=True,
                system_geometry=system_geometry,
                system_hamiltonian=ham,
            ),
            observables.Concurrence(
                site_index_from=tr_index,
                site_index_to=br_index,
                spin_up_from=True,
                spin_up_to=True,
                system_hamiltonian=ham,
                system_geometry=system_geometry,
                perform_checks=check_observable_imag,
                check_threshold=check_observable_imag_threshold,
            ),
            observables.Concurrence(
                site_index_from=bl_index,
                site_index_to=br_index,
                spin_up_from=True,
                spin_up_to=True,
                system_hamiltonian=ham,
                system_geometry=system_geometry,
                perform_checks=check_observable_imag,
                check_threshold=check_observable_imag_threshold,
            ),
            observables.Purity(
                site_index_from=tr_index,
                site_index_to=br_index,
                spin_up_from=True,
                spin_up_to=True,
                system_hamiltonian=ham,
                system_geometry=system_geometry,
                perform_checks=check_observable_imag,
                check_threshold=check_observable_imag_threshold,
            ),
            observables.Purity(
                site_index_from=bl_index,
                site_index_to=br_index,
                spin_up_from=True,
                spin_up_to=True,
                system_hamiltonian=ham,
                system_geometry=system_geometry,
                perform_checks=check_observable_imag,
                check_threshold=check_observable_imag_threshold,
            ),
        ]
        obs += obs_hard_coded

    else:
        raise Exception("Invalid arguments")

    if record_hamiltonian_properties:
        obs_ham_properties: List[observables.Observable] = []
        if isinstance(
            ham, hamiltonian.FirstOrderVariationalClassicalNetworkHamiltonian
        ):
            num_etas = ham.get_number_of_eta_parameters()
            num_base_params = ham.get_number_of_base_energy_parameters()
            # eta params of VCN
            for real_part_of_property in [True, False]:
                for eta_index in range(num_etas):
                    obs_ham_properties.append(
                        observables.VCNFactor(
                            ham=ham,
                            param_index=eta_index,
                            param_real_part=real_part_of_property,
                        )
                    )
            # variational U
            for real_part_of_property in [True, False]:
                obs_ham_properties.append(
                    observables.BaseEnergyFactor(
                        ham=ham,
                        param_index=-1,
                        param_real_part=real_part_of_property,
                        kind="U",
                    )
                )
            # variational eps
            for real_part_of_property in [True, False]:
                for base_param_index in range(num_base_params):
                    obs_ham_properties.append(
                        observables.BaseEnergyFactor(
                            ham=ham,
                            param_index=base_param_index,
                            param_real_part=real_part_of_property,
                            kind="eps",
                        )
                    )

        obs += obs_ham_properties

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
            record_imag_part=record_imag_part,
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
