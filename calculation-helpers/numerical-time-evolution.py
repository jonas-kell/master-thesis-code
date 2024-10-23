from typing import List, Tuple, Any, Dict, Union
import numpy as np
from scipy.linalg import expm
import os
import matplotlib.pyplot as plt
import threading
from partialTrace import partial_trace_out_b
from datetime import datetime
import json


def get_full_file_path(from_file_namne: str):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./../run-outputs/" + from_file_namne + ".json",
    )


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
        return np.cos(phiv) * (index % chain_length)

    def generate_basis(chain_length):
        basis = np.array(
            [state for state in np.ndindex(*(2 for _ in range(chain_length)))]
        )
        return basis

    # Example Hamiltonian for a linear chain of length n with 2 spin degrees
    basis = generate_basis(chain_length * 2)
    print("done generating basis")
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
        expectation_value_current_up = np.vdot(
            psi_t, np.dot(current_operator_up, psi_t)
        )
        expectation_value_current_down = np.vdot(
            psi_t, np.dot(current_operator_down, psi_t)
        )
        expectation_value_occupation = np.vdot(
            psi_t, np.dot(doube_occupation_first_site_operator, psi_t)
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
            expectation_value_current_up,
            expectation_value_current_down,
            expectation_value_occupation,
            expectation_value_purity,
            expectation_value_concurrence,
            expectation_value_concurrence,  # because obviously symm=asymm for our measurement, but not if the pauli measurements are taken wrong
        ]
        data.extend(sigma_measurements)

        measurements.append(
            {
                "time": t,
                "data": list(np.real(data)),
            }
        )

        print(f"Did time step {step_index+1} out of {number_of_time_steps}")

    observables = [
        {"type": "SpinCurrent", "label": "Current from site 0,1 up"},
        {"type": "SpinCurrent", "label": "Current from site 0,1 down"},
        {"type": "DoubleOccupationAtSite", "label": "Double occupation on site 0"},
        {"type": "Purity", "label": "Purity on site 0-1 up"},
        {"type": "Concurrence", "label": "Concurrence on site 0-1 up"},
        {"type": "ConcurrenceAsymm", "label": "Concurrence on site 0-1 up"},
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
    python_executable = "/bin/python3"
    arguments_string = (
        f'--file_name_overwrite "{file_name}" --U {U} --E {E} --J {J} --phi {phi} --start_time {start_time} --target_time_in_one_over_j {target_time_in_one_over_j} --number_of_time_steps {number_of_time_steps} --n {chain_length} --do_not_plot do_not_plot '
        + ("--number_workers 1 " if set_number_workers_to_one else "")
    )
    os.system(
        f"{python_executable} ./../computation-scripts/script.py {arguments_string}"
    )


def plot_experiment_comparison(
    filename_perturbation: str,
    filename_diagonalization: str,
    scaler_factor: float,
    scaler_factor_label: str = "J",
):
    loaded_data_perturbation: Dict[
        str, Union[float, str, Dict[Any, Any], List[Any]]
    ] = {}
    with open(get_full_file_path(filename_perturbation), mode="r") as file:
        loaded_data_perturbation = json.load(file)
    loaded_data_diagonalization: Dict[
        str, Union[float, str, Dict[Any, Any], List[Any]]
    ] = {}
    with open(get_full_file_path(filename_diagonalization), mode="r") as file:
        loaded_data_diagonalization = json.load(file)

    data_perturbation = loaded_data_perturbation["measurements"]
    data_diagonalization = loaded_data_diagonalization["measurements"]

    observables = loaded_data_diagonalization["observables"]

    for i, observable in enumerate(observables):
        label = observable["label"]

        times_to_plot_perturbation = []
        values_to_plot_perturbation = []
        times_to_plot_diagonalization = []
        values_to_plot_diagonalization = []

        for measurement_perturbation in data_perturbation:
            times_to_plot_perturbation.append(measurement_perturbation["time"])
            values_to_plot_perturbation.append(measurement_perturbation["data"][i])

        for measurement_diagonalization in data_diagonalization:
            times_to_plot_diagonalization.append(measurement_diagonalization["time"])
            values_to_plot_diagonalization.append(
                measurement_diagonalization["data"][i]
            )

        # Plotting
        plt.plot(
            np.array(times_to_plot_diagonalization) * scaler_factor,
            values_to_plot_diagonalization,
            label="Diagonalization",
        )
        plt.plot(
            np.array(times_to_plot_perturbation) * scaler_factor,
            values_to_plot_perturbation,
            label="Perturbation",
        )

        # Adding labels and title
        plt.xlabel("Time in 1/" + scaler_factor_label)
        plt.ylabel(label)
        plt.legend()

        # Show the plot
        plt.show()


def main():

    U: float = 1
    E: float = 0.5
    J: float = 0.1
    phi: float = np.pi / 10

    start_time: float = 0
    number_of_time_steps: int = 100
    target_time_in_one_over_j: float = 4

    chain_length = 2

    set_number_workers_to_one = True

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
            "target_time_in_one_over_j": target_time_in_one_over_j,
            "chain_length": chain_length,
            "set_number_workers_to_one": set_number_workers_to_one,
            "file_name": filename_for_main_thread,
        },
    )
    external_thread_perturbation.start()
    external_thread_diagonalization.start()

    external_thread_perturbation.join()
    external_thread_diagonalization.join()

    plot_experiment_comparison(
        filename_perturbation=filename_for_main_thread,
        filename_diagonalization=filename_for_diagonalization_thread,
        scaler_factor=np.abs(scaler_factor),
        scaler_factor_label=scaler_factor_label,
    )


if __name__ == "__main__":
    main()
