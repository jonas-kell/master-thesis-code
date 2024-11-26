from time import time as measure
import state
import systemgeometry
import numpy as np
import hamiltonian
import observables
from randomgenerator import RandomGenerator


def get_inefficient_4_way_flip_exp(
    use_state: state.SystemState,
    ham: hamiltonian.Hamiltonian,
    l: int,
    m: int,
    a: int,
    b: int,
    sigma_up: bool,
    mu_up: bool,
    measurement_time: float,
) -> np.complex128:
    modified_state = use_state.get_editable_copy()

    if sigma_up:
        use_l_index = l
        use_m_index = m
    else:
        use_l_index = use_state.get_opposite_spin_index(l)
        use_m_index = use_state.get_opposite_spin_index(m)
    if mu_up:
        use_a_index = a
        use_b_index = b
    else:
        use_a_index = use_state.get_opposite_spin_index(a)
        use_b_index = use_state.get_opposite_spin_index(b)

    modify_arr = modified_state.get_state_array()
    modify_arr[use_l_index] = 1 - modify_arr[use_l_index]
    modify_arr[use_m_index] = 1 - modify_arr[use_m_index]
    modify_arr[use_a_index] = 1 - modify_arr[use_a_index]
    modify_arr[use_b_index] = 1 - modify_arr[use_b_index]

    return np.exp(  # has the difference (modified)-(non-modified) already
        ham.get_H_eff_difference(
            time=measurement_time,
            system_state_a=modified_state,
            system_state_b=use_state,
        )
    )


def get_efficienter_4_way_flip_exp(
    use_state: state.SystemState,
    ham: hamiltonian.Hamiltonian,
    l: int,
    m: int,
    a: int,
    b: int,
    sigma_up: bool,
    mu_up: bool,
    measurement_time: float,
) -> np.complex128:
    if sigma_up:
        use_l_index = l
        use_m_index = m
    else:
        use_l_index = use_state.get_opposite_spin_index(l)
        use_m_index = use_state.get_opposite_spin_index(m)
    if mu_up:
        use_a_index = a
        use_b_index = b
    else:
        use_a_index = use_state.get_opposite_spin_index(a)
        use_b_index = use_state.get_opposite_spin_index(b)

    # this might never get called with use_l_index == use_a_index or use_m_index == use_b_index
    # that must be filtered out earlier
    if use_l_index == use_a_index or use_m_index == use_b_index:
        # check can be removed, when algo is correct (in real implementation)
        raise Exception("This should not happen")

    if use_l_index == use_b_index:  # use_m_index != use_a_index
        # also handles both equal cases here
        return np.exp(
            -ham.get_H_eff_difference_double_flipping(
                time=measurement_time,
                flipping1_up=sigma_up,
                flipping1_index=m,
                flipping2_up=mu_up,
                flipping2_index=a,
                before_swap_system_state=use_state,
            )[0]
        )
    elif use_m_index == use_a_index:  # use_l_index != use_b_index
        return np.exp(
            -ham.get_H_eff_difference_double_flipping(
                time=measurement_time,
                flipping1_up=sigma_up,
                flipping1_index=l,
                flipping2_up=mu_up,
                flipping2_index=b,
                before_swap_system_state=use_state,
            )[0]
        )
    else:
        # all use_indices use_l, use_m, use_a, use_b are different
        modify_arr = use_state.get_state_array()

        # N(4-flip) - N(no flip) = [N(4-flip) - N(2-flip)] + [N(no flip) - N(2-flip)]

        # flip in-place for computation
        modify_arr[use_l_index] = 1 - modify_arr[use_l_index]
        modify_arr[use_m_index] = 1 - modify_arr[use_m_index]
        a_result = ham.get_H_eff_difference_double_flipping(
            time=measurement_time,
            flipping1_up=mu_up,
            flipping1_index=a,
            flipping2_up=mu_up,
            flipping2_index=b,
            before_swap_system_state=use_state,
        )[0]
        # flip back, function doesn't modify state permanently, change is only in this section
        modify_arr[use_l_index] = 1 - modify_arr[use_l_index]
        modify_arr[use_m_index] = 1 - modify_arr[use_m_index]
        b_result = ham.get_H_eff_difference_double_flipping(
            time=measurement_time,
            flipping1_up=sigma_up,
            flipping1_index=l,
            flipping2_up=sigma_up,
            flipping2_index=m,
            before_swap_system_state=use_state,
        )[0]

        return np.exp(-a_result - b_result)


def eval_variance_op(
    use_state: state.SystemState,
    ham: hamiltonian.Hamiltonian,
    l: int,
    m: int,
    a: int,
    b: int,
    sigma_up: bool,
    mu_up: bool,
    measurement_time: float,
    four_way_flip_callback,
) -> np.complex128:
    if sigma_up:
        initial_occ_l = use_state.get_state_array()[l]
        initial_occ_m = use_state.get_state_array()[m]
    else:
        initial_occ_l = use_state.get_state_array()[
            use_state.get_opposite_spin_index(l)
        ]
        initial_occ_m = use_state.get_state_array()[
            use_state.get_opposite_spin_index(m)
        ]

    if mu_up:
        initial_occ_a = use_state.get_state_array()[a]
        initial_occ_b = use_state.get_state_array()[b]
    else:
        initial_occ_a = use_state.get_state_array()[
            use_state.get_opposite_spin_index(a)
        ]
        initial_occ_b = use_state.get_state_array()[
            use_state.get_opposite_spin_index(b)
        ]

    final_occ_l_square = initial_occ_l
    final_occ_m_square = initial_occ_m
    final_occ_a_square = initial_occ_a
    final_occ_b_square = initial_occ_b
    if sigma_up == mu_up:
        if l == a or m == a:
            final_occ_a_square = 1 - final_occ_a_square
        if l == b or m == b:
            final_occ_b_square = 1 - final_occ_b_square

    final_occ_l_double = initial_occ_l
    final_occ_m_double = initial_occ_m
    final_occ_a_double = initial_occ_a
    final_occ_b_double = initial_occ_b

    factor_square = (
        final_occ_l_square
        * (1 - final_occ_m_square)
        * final_occ_a_square
        * (1 - final_occ_b_square)
    )
    exp_JJ = 0
    if factor_square:
        exp_JJ = four_way_flip_callback(
            use_state=use_state,
            ham=ham,
            l=l,
            m=m,
            a=a,
            b=b,
            sigma_up=sigma_up,
            mu_up=mu_up,
            measurement_time=measurement_time,
        )
    factor_double = (
        final_occ_l_double
        * (1 - final_occ_m_double)
        * final_occ_a_double
        * (1 - final_occ_b_double)
    )
    exp_lm = 0
    exp_ab = 0
    if factor_double:
        exp_lm = np.exp(
            -ham.get_H_eff_difference_double_flipping(
                time=measurement_time,
                flipping1_up=sigma_up,
                flipping1_index=l,
                flipping2_up=sigma_up,
                flipping2_index=m,
                before_swap_system_state=use_state,
            )[0]
        )
        exp_ab = np.exp(
            -ham.get_H_eff_difference_double_flipping(
                time=measurement_time,
                flipping1_up=mu_up,
                flipping1_index=a,
                flipping2_up=mu_up,
                flipping2_index=b,
                before_swap_system_state=use_state,
            )[0]
        )

    return ham.J * ham.J * ((exp_JJ) - (exp_lm * exp_ab))


def main():
    U = 0.3
    E = -0.5
    J = 0.2
    phi = np.pi / 3
    measurement_time = 1.2
    n = 6

    random = RandomGenerator(str(measure()))

    system_geometry = systemgeometry.LinearChainNonPeriodicState(n)
    system_geometry.init_index_overlap_circle_cache(2)

    initial_system_state = state.HomogenousInitialSystemState(system_geometry)

    use_hamiltonian = hamiltonian.HardcoreBosonicHamiltonianFlippingAndSwappingOptimizationSecondOrder(
        U=U,
        E=E,
        J=J,
        phi=phi,
        initial_system_state=initial_system_state,
        system_geometry=system_geometry,
    )
    # use_hamiltonian = (
    #     hamiltonian.HardcoreBosonicHamiltonianFlippingAndSwappingOptimization(
    #         U=U,
    #         E=E,
    #         J=J,
    #         phi=phi,
    #         initial_system_state=initial_system_state,
    #     )
    # )
    # use_hamiltonian = hamiltonian.ZerothOrderFlippingAndSwappingOptimization(
    #     U=U,
    #     E=E,
    #     J=J,
    #     phi=phi,
    #     initial_system_state=initial_system_state,
    # )

    use_state = state.SystemState(system_geometry, initial_system_state)
    observable_var = observables.EnergyVariance(
        ham=use_hamiltonian, geometry=system_geometry
    )

    total_time_all_aggregation = 0
    total_time_only_overlap_aggregation = 0
    total_time_only_overlap_aggregation_eff_flip = 0
    total_time_implementation = 0
    iterations = 1
    for iter_index in range(iterations):
        use_state.init_random_filling(random)

        start = measure() * 1000
        all_aggregator = np.complex128(0)
        for sigma_up in [True, False]:
            for mu_up in [True, False]:
                num_all_index_pairs = 0
                for l in range(system_geometry.get_number_sites_wo_spin_degree()):
                    for m in system_geometry.get_nearest_neighbor_indices(l):
                        for a in range(
                            system_geometry.get_number_sites_wo_spin_degree()
                        ):
                            for b in system_geometry.get_nearest_neighbor_indices(a):
                                num_all_index_pairs += 1
                                all_aggregator += eval_variance_op(
                                    ham=use_hamiltonian,
                                    use_state=use_state,
                                    l=l,
                                    m=m,
                                    a=a,
                                    b=b,
                                    sigma_up=sigma_up,
                                    mu_up=mu_up,
                                    measurement_time=measurement_time,
                                    four_way_flip_callback=get_inefficient_4_way_flip_exp,
                                )
                if iter_index == 0 and sigma_up and mu_up:
                    print(f"Full convolution took {num_all_index_pairs} index tuples")
        end = measure() * 1000
        total_time_all_aggregation += end - start

        start = measure() * 1000
        only_overlap_aggregator = np.complex128(0)
        for sigma_up in [True, False]:
            for mu_up in [True, False]:
                for l, m, a, b in system_geometry.get_index_overlap_circle_tuples():
                    only_overlap_aggregator += eval_variance_op(
                        ham=use_hamiltonian,
                        use_state=use_state,
                        l=l,
                        m=m,
                        a=a,
                        b=b,
                        sigma_up=sigma_up,
                        mu_up=mu_up,
                        measurement_time=measurement_time,
                        four_way_flip_callback=get_inefficient_4_way_flip_exp,
                    )
        end = measure() * 1000
        total_time_only_overlap_aggregation += end - start

        start = measure() * 1000
        only_overlap_aggregator_eff_flip = np.complex128(0)
        for sigma_up in [True, False]:
            for mu_up in [True, False]:
                for l, m, a, b in system_geometry.get_index_overlap_circle_tuples():
                    only_overlap_aggregator_eff_flip += eval_variance_op(
                        ham=use_hamiltonian,
                        use_state=use_state,
                        l=l,
                        m=m,
                        a=a,
                        b=b,
                        sigma_up=sigma_up,
                        mu_up=mu_up,
                        measurement_time=measurement_time,
                        four_way_flip_callback=get_efficienter_4_way_flip_exp,
                    )
        end = measure() * 1000
        total_time_only_overlap_aggregation_eff_flip += end - start

        start = measure() * 1000
        result_implementation = (
            observable_var.get_expectation_value(
                time=measurement_time, system_state=use_state
            )  # needs renormalization away from number of sites
            * system_geometry.get_number_sites_wo_spin_degree()
        )
        end = measure() * 1000
        total_time_implementation += end - start

        if np.abs(all_aggregator - only_overlap_aggregator) > 1e-6:
            print(all_aggregator)
            print(only_overlap_aggregator)
            raise Exception("Should be the same")

        if np.abs(all_aggregator - only_overlap_aggregator_eff_flip) > 1e-6:
            print(all_aggregator)
            print(only_overlap_aggregator_eff_flip)
            raise Exception("Should be the same")

        if np.abs(all_aggregator - result_implementation) > 1e-6:
            print(all_aggregator)
            print(result_implementation)
            raise Exception("Should be the same finally")

    print("All aggregation ms:", total_time_all_aggregation)
    print("Overlap aggregation ms:", total_time_only_overlap_aggregation)
    print(
        "Overlap aggregation better flipping:",
        total_time_only_overlap_aggregation_eff_flip,
    )
    print(
        "Final implementation:",
        total_time_implementation,
    )


if __name__ == "__main__":
    main()
