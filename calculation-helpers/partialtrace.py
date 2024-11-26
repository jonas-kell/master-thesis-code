import numpy as np


def partial_trace_out_b(rho, no_qubits_a, no_qubits_b):
    dim_A = 2**no_qubits_a
    dim_B = 2**no_qubits_b
    rho_reshaped = np.reshape(rho, [dim_A, dim_B, dim_A, dim_B])
    reduced_rho = np.trace(rho_reshaped, axis1=1, axis2=3)

    return reduced_rho


def generate_basis(chain_length):
    basis = np.array([state for state in np.ndindex(*(2 for _ in range(chain_length)))])
    return basis


def generate_random_density_matrix(num_q_bits: int):
    psi = np.random.randn(2**num_q_bits) + 1j * np.random.randn(2**num_q_bits)
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())

    return rho


def find_index_of_basis(basis, comparison):
    return np.where(np.all(basis == comparison, axis=1))[0][0]


def main():
    num_qubits_to_keep_after_tracout = 2
    num_qubits_to_trace_out = 1

    num_qubits_total = num_qubits_to_keep_after_tracout + num_qubits_to_trace_out

    density_matrix = generate_random_density_matrix(num_qubits_total)
    # print(density_matrix)
    print(np.trace(density_matrix))

    full_basis = generate_basis(num_qubits_total)
    keep_basis = generate_basis(num_qubits_to_keep_after_tracout)
    trace_out_basis = generate_basis(num_qubits_to_trace_out)

    # trace out everything except first two qubits manually
    # https://www.youtube.com/watch?v=L70TVZHYOsM
    res_traced_out_matrix = np.zeros(
        (num_qubits_to_keep_after_tracout**2, num_qubits_to_keep_after_tracout**2),
        dtype=np.complex128,
    )
    for row_keep_index, row_state_keep in enumerate(keep_basis):
        for col_keep_index, col_state_keep in enumerate(keep_basis):
            for trace_out_elem in trace_out_basis:
                sum_row_index = find_index_of_basis(
                    full_basis, np.concatenate((row_state_keep, trace_out_elem))
                )
                sum_col_index = find_index_of_basis(
                    full_basis, np.concatenate((col_state_keep, trace_out_elem))
                )
                res_traced_out_matrix[row_keep_index][col_keep_index] += density_matrix[
                    sum_row_index
                ][sum_col_index]
    print(np.trace(res_traced_out_matrix))

    direct_trace_out = partial_trace_out_b(
        density_matrix, num_qubits_to_keep_after_tracout, num_qubits_to_trace_out
    )

    print("comparison")

    print(res_traced_out_matrix)
    print(direct_trace_out)
    print(np.abs(res_traced_out_matrix - direct_trace_out) < 1e-4)


if __name__ == "__main__":
    main()
