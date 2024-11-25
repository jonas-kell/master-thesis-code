from time import time as measure
import state
import systemgeometry
import numpy as np
import hamiltonian
import observables
from randomgenerator import RandomGenerator


def main():
    U = 0.3
    E = -0.5
    J = 1.1
    phi = np.pi / 3
    measurement_time = 1.2
    n = 6

    random = RandomGenerator(str(measure()))

    system_geometry = systemgeometry.SquareSystemNonPeriodicState(n)

    initial_system_state = state.HomogenousInitialSystemState(system_geometry)

    use_hamiltonian = hamiltonian.HardcoreBosonicHamiltonianFlippingAndSwappingOptimizationSecondOrder(
        U=U,
        E=E,
        J=J,
        phi=phi,
        initial_system_state=initial_system_state,
        system_geometry=system_geometry,
    )
    from_index = n // 2
    to_index = system_geometry.get_nearest_neighbor_indices(from_index)[0]
    spin_up = True
    old_current_operator = observables.SpinCurrent(
        site_index_from=from_index,
        site_index_to=to_index,
        spin_up=spin_up,
        system_hamiltonian=use_hamiltonian,
        system_geometry=system_geometry,
    )
    new_current_operator = observables.SpinCurrentFlipping(
        site_index_from=from_index,
        site_index_to=to_index,
        spin_up=spin_up,
        system_hamiltonian=use_hamiltonian,
        system_geometry=system_geometry,
    )
    # concurrence_operator = observables.Concurrence(
    #     site_index_from=from_index,
    #     spin_up_from=spin_up,
    #     site_index_to=to_index,
    #     spin_up_to=spin_up,
    #     system_hamiltonian=use_hamiltonian,
    #     system_geometry=system_geometry,
    # )
    # concurrence_asymm_operator = observables.ConcurrenceAsymm(
    #     site_index_from=from_index,
    #     spin_up_from=spin_up,
    #     site_index_to=to_index,
    #     spin_up_to=spin_up,
    #     system_hamiltonian=use_hamiltonian,
    #     system_geometry=system_geometry,
    # )

    use_state = state.SystemState(system_geometry, initial_system_state)

    total_time_old_current = 0
    total_time_new_current = 0
    # total_time_concurrence = 0
    # total_time_concurrence_asymm = 0
    iterations = 3000
    for _ in range(iterations):
        use_state.init_random_filling(random)

        start = measure() * 1000
        old_current = old_current_operator.get_expectation_value(
            time=measurement_time, system_state=use_state
        )
        end = measure() * 1000
        total_time_old_current += end - start

        start = measure() * 1000
        new_current = new_current_operator.get_expectation_value(
            time=measurement_time, system_state=use_state
        )
        end = measure() * 1000
        total_time_new_current += end - start

        if np.abs(old_current - new_current) / (np.abs(old_current) + 1e-10) > 1e-10:
            print(old_current)
            print(new_current)
            raise Exception("Should be the same")

        # start = measure() * 1000
        # concurrence = concurrence_operator.post_process(
        #     concurrence_operator.get_expectation_value(
        #         time=measurement_time, system_state=use_state
        #     )
        # )
        # end = measure() * 1000
        # total_time_concurrence += end - start

        # start = measure() * 1000
        # concurrence_asymm = concurrence_asymm_operator.post_process(
        #     concurrence_asymm_operator.get_expectation_value(
        #         time=measurement_time, system_state=use_state
        #     )
        # )
        # end = measure() * 1000
        # total_time_concurrence_asymm += end - start

        # if np.abs(concurrence - concurrence_asymm) > 1e-6:
        #     print(concurrence)
        #     print(concurrence_asymm)
        #     raise Exception("Should be the same")

    print("Current old ms:", total_time_old_current)
    print("Current new ms:", total_time_new_current)
    # print("Concurrence ms:", total_time_concurrence)
    # print("Concurrence asym ms:", total_time_concurrence_asymm)


if __name__ == "__main__":
    main()
