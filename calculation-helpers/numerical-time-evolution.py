import numpy as np
from scipy.linalg import expm
import os
import matplotlib.pyplot as plt
import threading


def numerically_calculate_time_evolution(
    U: float = 1,
    E: float = -1,
    J: float = 0.1,
    phi: float = np.pi / 10,
    start_time: float = 0,
    number_of_time_steps: int = 10,
    time_step: float = 1,
    chain_length: int = 2,
):
    def compare_bra_ket(bra, ket):
        return np.all(bra == ket)

    # does c#_l c_m
    def compare_m_to_l_hopped_ket_to_bra(bra, ket, l, m):
        if ket[m] == 0:
            return False
        if ket[l] == 1:
            return False

        copy_of_ket = np.copy(ket)

        copy_of_ket[m] = 0
        copy_of_ket[l] = 1

        return compare_bra_ket(bra, copy_of_ket)

    def get_eps_multiplier(index: int, phiv: float, chain_length: int) -> float:
        return np.cos(phiv) * (index % chain_length)

    def generate_basis(chain_length):
        basis = np.array(
            [state for state in np.ndindex(*(2 for _ in range(chain_length * 2)))]
        )
        return basis

    # Example Hamiltonian for a linear chain of length n
    basis = generate_basis(chain_length)
    H = np.array(
        [
            [
                +U
                * np.sum(
                    np.array(
                        [
                            (
                                1
                                * compare_bra_ket(bra_state, ket_state)
                                * bra_state[index]
                                * bra_state[index + chain_length]
                            )
                            for index in range(chain_length)
                        ]
                    )
                )
                + E
                * np.sum(
                    np.array(
                        [
                            (
                                1
                                * get_eps_multiplier(index, phi, chain_length)
                                * compare_bra_ket(bra_state, ket_state)
                                * bra_state[index]
                            )
                            for index in range(chain_length * 2)
                        ]
                    )
                )
                - J
                * np.sum(
                    np.array(
                        [
                            (
                                1
                                * compare_m_to_l_hopped_ket_to_bra(
                                    bra_state, ket_state, ind1, ind2
                                )
                                + 1
                                * compare_m_to_l_hopped_ket_to_bra(
                                    bra_state, ket_state, ind2, ind1
                                )
                            )
                            for (ind1, ind2) in (
                                [
                                    (index, index + 1)
                                    for index in range(chain_length - 1)
                                ]
                                + [
                                    (chain_length + index, chain_length + index + 1)
                                    for index in range(chain_length - 1)
                                ]
                            )
                        ]
                    )
                )
                for ket_state in basis
            ]
            for bra_state in basis
        ]
    )
    if not np.all(np.conjugate(H.T) == H):
        print("H not hermetian")
    # print(H)

    # Initial state
    d = 2 ** (chain_length * 2)
    # Dimension of the Hilbert space 2 spin degrees on n particles
    amplitude = 1 / np.sqrt(d)  # Same amplitude for all basis states
    psi_0 = np.full(d, amplitude, dtype=complex)
    # print(psi_0)

    # Observable current operator from site 0 to 1 on spin up
    current_op_from = 0
    current_op_to = 1
    current_operator = np.array(
        [
            [
                (
                    1j
                    * compare_m_to_l_hopped_ket_to_bra(
                        bra_state, ket_state, current_op_from, current_op_to
                    )
                    - 1j
                    * compare_m_to_l_hopped_ket_to_bra(
                        bra_state, ket_state, current_op_to, current_op_from
                    )
                )
                for ket_state in basis
            ]
            for bra_state in basis
        ]
    )
    # print(current_operator)
    at_site = 0
    doube_occupation_first_site_operator = np.array(
        [
            [
                (
                    compare_bra_ket(bra_state, ket_state)
                    * ket_state[at_site]
                    * ket_state[at_site + chain_length]
                )
                for ket_state in basis
            ]
            for bra_state in basis
        ]
    )
    # print(doube_occupation_first_site_operator)

    time_values = []
    expectation_values_current = []
    expectation_values_occupation = []
    for step_index in range(number_of_time_steps):
        t = start_time + step_index * time_step

        # Compute the matrix exponential for time evolution operator
        U_t = expm(-1j * H * t)

        # Time evolved state
        psi_t = np.dot(U_t, psi_0)
        if np.abs(np.sum(np.square(np.abs(psi_t))) - 1) > 1e-2:
            print("No longer normalized time evolved state")

        # expectation value
        expectation_value_current = np.vdot(psi_t, np.dot(current_operator, psi_t))
        expectation_value_occupation = np.vdot(
            psi_t, np.dot(doube_occupation_first_site_operator, psi_t)
        )

        time_values.append(t)
        expectation_values_current.append(expectation_value_current)
        expectation_values_occupation.append(expectation_value_occupation)

    return (time_values, expectation_values_current, expectation_values_occupation)


def run_main_program(
    U: float = 1,
    E: float = -1,
    J: float = 0.1,
    phi: float = np.pi / 10,
    start_time: float = 0,
    number_of_time_steps: int = 10,
    target_time_in_one_over_j: float = 8,
    chain_length: int = 2,
):
    python_executable = "python"
    arguments_string = f"--U {U} --E {E} --J {J} --phi {phi} --start_time {start_time} --target_time_in_one_over_j {target_time_in_one_over_j} --number_of_time_steps {number_of_time_steps} --n {chain_length}"
    os.system(
        f"{python_executable} ./../computation-scripts/script.py {arguments_string}"
    )


def plot_experiment(
    times,
    values_current,
    values_occupation,
    scaler_factor: float,
    scaler_factor_label: str = "J",
):
    # Plotting
    plt.plot(np.array(times) * scaler_factor, values_current, label="Current")
    plt.plot(
        np.array(times) * scaler_factor, values_occupation, label="Double Occupation 0"
    )

    # Adding labels and title
    plt.xlabel("Time in 1/" + scaler_factor_label)
    plt.ylabel("Values")
    plt.legend()

    # Show the plot
    plt.show()


def main():

    U: float = 1
    E: float = 0.5
    J: float = 0.1
    phi: float = np.pi / 10

    start_time: float = 0
    number_of_time_steps: int = 400
    target_time_in_one_over_j: float = 4

    chain_length = 2

    # computed
    if np.abs(J) < 1e-5:
        # if J interaction deactivated, scale with U
        scaler_factor = U
        scaler_factor_label = "U"
    else:
        scaler_factor = J
        scaler_factor_label = "J"
    target_time: float = (1 / np.abs(scaler_factor)) * target_time_in_one_over_j
    time_step: float = (target_time - start_time) / number_of_time_steps

    (
        numerical_time_values,
        numerical_expectation_values_current,
        numerical_expectation_values_occupation,
    ) = numerically_calculate_time_evolution(
        U=U,
        E=E,
        J=J,
        phi=phi,
        start_time=start_time,
        number_of_time_steps=number_of_time_steps,
        time_step=time_step,
        chain_length=chain_length,
    )

    external_thread = threading.Thread(
        target=run_main_program,
        args={
            U: U,
            E: E,
            J: J,
            phi: phi,
            start_time: start_time,
            number_of_time_steps: number_of_time_steps,
            target_time_in_one_over_j: target_time_in_one_over_j,
            chain_length: chain_length,
        },
    )
    external_thread.start()

    print(numerical_expectation_values_current)
    print(numerical_expectation_values_occupation)
    plot_experiment(
        numerical_time_values,
        numerical_expectation_values_current,
        numerical_expectation_values_occupation,
        np.abs(scaler_factor),
        scaler_factor_label,
    )


if __name__ == "__main__":
    main()
