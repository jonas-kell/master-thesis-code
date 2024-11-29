import numpy as np
from scipy.linalg import expm
import os
import threading
from partialtrace import partial_trace_out_b
from datetime import datetime
import json
from commonsettings import get_full_file_path, scaler_switch
from plotcompare import plot_experiment_comparison


def matrix_sqrt(matr: np.ndarray) -> np.ndarray:
    evalues, evectors = np.linalg.eigh(matr)
    sqrt_matrix = evectors * np.emath.sqrt(evalues) @ np.linalg.inv(evectors)
    return sqrt_matrix


sigmay = np.array([[0, -1j], [1j, 0]])
sigmaysigmay = np.array(
    [
        [sigmay[row // 2][col // 2] * sigmay[row % 2][col % 2] for col in range(4)]
        for row in range(4)
    ]
)


def calculate_concurrence(rho_reduced_in_occ_basis) -> np.complex128:
    spin_flipped = sigmaysigmay @ np.conjugate(rho_reduced_in_occ_basis) @ sigmaysigmay
    sqrt_rho = matrix_sqrt(rho_reduced_in_occ_basis)
    R_matrix = matrix_sqrt(sqrt_rho @ spin_flipped @ sqrt_rho)

    # R was checked to be hermitian in a test run here

    eigenvals = np.flip(np.linalg.eigvalsh(R_matrix))
    return np.max([0, eigenvals[0] - eigenvals[1] - eigenvals[2] - eigenvals[3]])


def get_rho_and_spin_measurements(rho_reduced):
    # expectation values and parts that make up the density matrix

    sigma_one = np.array([[1, 0], [0, 1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    def combine_operators(op_a, op_b) -> np.ndarray:
        return np.array(
            [
                [op_a[row // 2][col // 2] * op_b[row % 2][col % 2] for col in range(4)]
                for row in range(4)
            ],
            dtype=np.complex64,
        )

    operators = [sigma_one, sigma_x, sigma_y, sigma_z]

    sigma_measurements = [
        [
            np.trace(rho_reduced @ combine_operators(op_row, op_col))
            for op_col in operators
        ]
        for op_row in operators
    ]

    return np.array(sigma_measurements).flatten()


def numerically_calculate_time_evolution(
    U: float = 1,
    E: float = -1,
    J: float = 0.1,
    phi: float = np.pi / 10,
    start_time: float = 0,
    number_of_time_steps: int = 10,
    time_step: float = 1,
    chain_length: int = 2,
    file_name="measurement_for_diagonalization",
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
        # Attempt to check the lateral symmetry of flipping, by placing the 0-index in the "middle" to make it E-symmetric
        # this should not influence other properties (I think)
        return np.cos(phiv) * ((index % chain_length) - (chain_length / 2.0) + 0.5)

    def generate_basis(chain_length):
        # generate with spins 0,1up easily extractable by tracing out to reduced density matrix
        # starting with up,up,up,up.... to have canonical ordering for sigma_y convention

        # make sure, to start the sampling
        # from 1,1,1,1,1
        # then 1,1,1,1,0
        # then 1,1,1,0,1
        # ....
        # last 0,0,0,0,0

        basis = 1 - np.array(
            [state for state in np.ndindex(*(2 for _ in range(chain_length)))]
        )
        return basis

    # Example Hamiltonian for a linear chain of length n with 2 spin degrees
    basis = generate_basis(chain_length * 2)
    print("done generating basis")
    H_0 = np.array(
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
                for ket_state in basis
            ]
            for bra_state in basis
        ]
    )
    V = np.array(
        [
            [
                -J
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
    H = H_0 + V
    print("done generating hamiltonian")
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

    def get_current_operator(from_index, to_index, up: bool = True):
        return -J * np.array(
            [
                [
                    (
                        1j
                        * compare_m_to_l_hopped_ket_to_bra(
                            bra_state,
                            ket_state,
                            from_index + (0 if up else chain_length),
                            to_index + (0 if up else chain_length),
                        )
                        - 1j
                        * compare_m_to_l_hopped_ket_to_bra(
                            bra_state,
                            ket_state,
                            to_index + (0 if up else chain_length),
                            from_index + (0 if up else chain_length),
                        )
                    )
                    for ket_state in basis
                ]
                for bra_state in basis
            ]
        )

    current_operator_up = get_current_operator(current_op_from, current_op_to, True)
    current_operator_down = get_current_operator(current_op_from, current_op_to, False)
    # print(current_operator)

    def get_double_occupation_operator(at_site: int):
        return np.array(
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

    doube_occupation_zero_site_operator = get_double_occupation_operator(0)
    doube_occupation_one_site_operator = get_double_occupation_operator(1)

    # print(doube_occupation_first_site_operator)
    def get_occupation_operator(index: int, up: bool = True):
        index_to_use_for_occ = index
        if not up:
            index_to_use_for_occ += chain_length

        return np.array(
            [
                [
                    (
                        compare_bra_ket(bra_state, ket_state)
                        * ket_state[index_to_use_for_occ]
                    )
                    for ket_state in basis
                ]
                for bra_state in basis
            ]
        )

    zero_up_occupation_operator = get_occupation_operator(0, True)
    zero_down_occupation_operator = get_occupation_operator(0, False)
    one_up_occupation_operator = get_occupation_operator(1, True)
    one_down_occupation_operator = get_occupation_operator(1, False)
    H_squared_operator = H @ H
    V_squared_operator = V @ V
    H0_squared_operator = H_0 @ H_0
    H0_V_operator = H_0 @ V

    measurements = []
    for step_index in range(number_of_time_steps):
        t = start_time + step_index * time_step

        # Compute the matrix exponential for time evolution operator
        U_t = expm(-1j * H * t)

        # Time evolved state
        psi_t = np.dot(U_t, psi_0)
        if np.abs(np.sum(np.square(np.abs(psi_t))) - 1) > 1e-2:
            print("No longer normalized time evolved state")

        # expectation value
        expectation_value_energy = np.vdot(psi_t, np.dot(H, psi_t)) / chain_length
        expectation_value_energy_variance = (
            np.vdot(psi_t, np.dot(H_squared_operator, psi_t))
            - (expectation_value_energy * chain_length) ** 2
        ) / chain_length
        expectation_value_energy_variance_2 = (
            np.vdot(psi_t, np.dot(V_squared_operator, psi_t))
            - np.vdot(psi_t, np.dot(V, psi_t)) ** 2
        ) / chain_length
        expectation_value_energy_variance_3 = (
            expectation_value_energy_variance_2 * chain_length
            + np.vdot(psi_t, np.dot(H0_squared_operator, psi_t))
            - np.vdot(psi_t, np.dot(H_0, psi_t)) ** 2
            + 2
            * (
                np.vdot(psi_t, np.dot(H0_V_operator, psi_t))
                - np.vdot(psi_t, np.dot(H_0, psi_t)) * np.vdot(psi_t, np.dot(V, psi_t))
            )
        ) / chain_length
        expectation_value_current_up = np.vdot(
            psi_t, np.dot(current_operator_up, psi_t)
        )
        expectation_value_current_down = np.vdot(
            psi_t, np.dot(current_operator_down, psi_t)
        )
        expectation_value_zero_double_occupation = np.vdot(
            psi_t, np.dot(doube_occupation_zero_site_operator, psi_t)
        )
        expectation_value_one_double_occupation = np.vdot(
            psi_t, np.dot(doube_occupation_one_site_operator, psi_t)
        )
        expectation_value_zero_up_occupation = np.vdot(
            psi_t, np.dot(zero_up_occupation_operator, psi_t)
        )
        expectation_value_zero_down_occupation = np.vdot(
            psi_t, np.dot(zero_down_occupation_operator, psi_t)
        )
        expectation_value_one_up_occupation = np.vdot(
            psi_t, np.dot(one_up_occupation_operator, psi_t)
        )
        expectation_value_one_down_occupation = np.vdot(
            psi_t, np.dot(one_down_occupation_operator, psi_t)
        )

        # concurrence
        rho = np.outer(psi_t, np.conjugate(psi_t))  # Density matrix = |psi><psi|
        if np.abs(np.trace(rho) - 1) > 1e-5:
            print(rho)
            print("Density matrix is no longer of trace 1")

        rho_reduced = partial_trace_out_b(rho, 2, chain_length * 2 - 2)
        if np.abs(np.trace(rho_reduced) - 1) > 1e-5:
            print(rho_reduced)
            print("Reduced Density matrix is no longer of trace 1")

        sigma_measurements = get_rho_and_spin_measurements(rho_reduced)

        expectation_value_purity = np.trace(rho_reduced @ rho_reduced)

        expectation_value_concurrence = calculate_concurrence(rho_reduced)

        data = [
            # expectation_value_energy,
            # expectation_value_energy_variance,
            # expectation_value_energy_variance_2,
            # expectation_value_energy_variance_3,
            expectation_value_concurrence,
            expectation_value_concurrence,  # because obviously symm=asymm for our measurement, but not if the pauli measurements are taken wrong
            expectation_value_purity,
            # expectation_value_current_up,
            # expectation_value_current_down,
            # expectation_value_zero_double_occupation,
            # expectation_value_one_double_occupation,
            # expectation_value_zero_up_occupation,
            # expectation_value_zero_down_occupation,
            # expectation_value_one_up_occupation,
            # expectation_value_one_down_occupation,
        ]
        data.extend(sigma_measurements)

        measurements.append(
            {
                "time": t,
                "data": list(np.real(data)),
            }
        )

        #! extract probabilities during run #TODO remove
        # if t > 390:
        #     print("diagonalization")

        #     for state_index, state in enumerate(basis):
        #         print(state)
        #         print(np.square(np.abs(psi_t[state_index])))

        #     exit()

        print(
            f"Did time step {step_index+1} out of {number_of_time_steps} at time {t:.3f}"
        )

    observables = [
        # {"type": "Energy", "label": "Energy per site"},
        # {"type": "EnergyVariance", "label": "Energy Variance per site"},
        # {"type": "EnergyVariance", "label": "V only - Energy Variance per site"},
        # {"type": "EnergyVariance", "label": "Composite - Energy Variance per site"},
        {"type": "Concurrence", "label": "Concurrence on site 0-1 up"},
        {"type": "ConcurrenceAsymm", "label": "Concurrence on site 0-1 up"},
        {"type": "Purity", "label": "Purity on site 0-1 up"},
        # {"type": "SpinCurrent", "label": "Current from site 0,1 up"},
        # {"type": "SpinCurrent", "label": "Current from site 0,1 down"},
        # {"type": "DoubleOccupationAtSite", "label": "Double occupation on site 0"},
        # {"type": "DoubleOccupationAtSite", "label": "Double occupation on site 1"},
        # {"type": "OccupationAtSite", "label": "Occupation on site 0, up"},
        # {"type": "OccupationAtSite", "label": "Occupation on site 0, down"},
        # {"type": "OccupationAtSite", "label": "Occupation on site 1, up"},
        # {"type": "OccupationAtSite", "label": "Occupation on site 1, down"},
    ]
    observables.extend(
        [
            {"type": "PauliMeasurement", "label": "Pauli 00"},
            {"type": "PauliMeasurement", "label": "Pauli 0x"},
            {"type": "PauliMeasurement", "label": "Pauli 0y"},
            {"type": "PauliMeasurement", "label": "Pauli 0z"},
            {"type": "PauliMeasurement", "label": "Pauli x0"},
            {"type": "PauliMeasurement", "label": "Pauli xx"},
            {"type": "PauliMeasurement", "label": "Pauli xy"},
            {"type": "PauliMeasurement", "label": "Pauli xz"},
            {"type": "PauliMeasurement", "label": "Pauli y0"},
            {"type": "PauliMeasurement", "label": "Pauli yx"},
            {"type": "PauliMeasurement", "label": "Pauli yy"},
            {"type": "PauliMeasurement", "label": "Pauli yz"},
            {"type": "PauliMeasurement", "label": "Pauli z0"},
            {"type": "PauliMeasurement", "label": "Pauli zx"},
            {"type": "PauliMeasurement", "label": "Pauli zy"},
            {"type": "PauliMeasurement", "label": "Pauli zz"},
        ]
    )

    with open(get_full_file_path(file_name), mode="w", newline="") as file:
        json.dump(
            {
                "hamiltonian": {
                    "U": U,
                    "E": E,
                    "J": J,
                    "phi": phi,
                    "type": "ExactDiagonalization",
                },
                "start_time": start_time,
                "time_step": time_step,
                "number_of_time_steps": number_of_time_steps,
                "observables": observables,
                "measurements": measurements,
            },
            file,
        )


def run_main_program(
    U: float = 1,
    E: float = -1,
    J: float = 0.1,
    phi: float = np.pi / 10,
    start_time: float = 0,
    number_of_time_steps: int = 10,
    target_time_in_one_over_j: float = 8,
    chain_length: int = 2,
    set_number_workers_to_one: bool = True,
    file_name="measurement_for_numerical",
):
    python_executable = "python"
    arguments_string = (
        f'--file_name_overwrite "{file_name}" --U {U} --E {E} --J {J} --phi {phi} --start_time {start_time} --target_time_in_one_over_j {target_time_in_one_over_j} --number_of_time_steps {number_of_time_steps} --n {chain_length} --do_not_plot do_not_plot '
        + ("--number_workers 1 " if set_number_workers_to_one else "")
    )
    os.system(
        f"{python_executable} ./../computation-scripts/script.py {arguments_string}"
    )


def main():

    U: float = 1
    E: float = 0.5
    J: float = 0.1
    phi: float = np.pi / 10

    start_time: float = 0
    number_of_time_steps: int = 200
    target_time_in_one_over_scaler: float = 10

    chain_length = 4

    set_number_workers_to_one = False

    if np.abs(J) < scaler_switch:
        # if J interaction "deactivated", scale with U
        scaler_factor = U
        main_program_j_input_rescaler = J / U
    else:
        scaler_factor = J
        main_program_j_input_rescaler = 1

    target_time: float = (1 / np.abs(scaler_factor)) * target_time_in_one_over_scaler
    time_step: float = (target_time - start_time) / number_of_time_steps

    time_string = datetime.now().strftime("%Y-%m-%d__%H,%M,%S")
    filename_for_main_thread = "perturbation_measurements_" + time_string
    filename_for_diagonalization_thread = "diagonalization_measurements_" + time_string

    external_thread_diagonalization = threading.Thread(
        target=numerically_calculate_time_evolution,
        kwargs={
            "U": U,
            "E": E,
            "J": J,
            "phi": phi,
            "start_time": start_time,
            "number_of_time_steps": number_of_time_steps,
            "time_step": time_step,
            "chain_length": chain_length,
            "file_name": filename_for_diagonalization_thread,
        },
    )
    external_thread_perturbation = threading.Thread(
        target=run_main_program,
        kwargs={
            "U": U,
            "E": E,
            "J": J,
            "phi": phi,
            "start_time": start_time,
            "number_of_time_steps": number_of_time_steps,
            "target_time_in_one_over_j": target_time_in_one_over_scaler
            * main_program_j_input_rescaler,
            "chain_length": chain_length,
            "set_number_workers_to_one": set_number_workers_to_one,
            "file_name": filename_for_main_thread,
        },
    )
    external_thread_perturbation.start()
    external_thread_diagonalization.start()

    external_thread_perturbation.join()
    external_thread_diagonalization.join()

    print(filename_for_main_thread, filename_for_diagonalization_thread)
    plot_experiment_comparison(
        [filename_for_diagonalization_thread, filename_for_main_thread],
        ["Diagonalization", "Perturbation"],
    )


if __name__ == "__main__":
    main()
