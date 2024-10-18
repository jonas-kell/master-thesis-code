import numpy as np

# from "spinToZBasisTransformation.py"
density_matrix_builder_helper = np.array(
    [
        [
            [
                [(1 + 0j), 0j, 0j, 0j],
                [0j, (1 + 0j), 0j, 0j],
                [0j, 0j, (1 + 0j), 0j],
                [0j, 0j, 0j, (1 + 0j)],
            ],
            [
                [0j, (1 + 0j), 0j, 0j],
                [(1 + 0j), 0j, 0j, 0j],
                [0j, 0j, 0j, (1 + 0j)],
                [0j, 0j, (1 + 0j), 0j],
            ],
            [
                [0j, -1j, 0j, 0j],
                [1j, 0j, 0j, 0j],
                [0j, 0j, 0j, -1j],
                [0j, 0j, 1j, 0j],
            ],
            [
                [(1 + 0j), 0j, 0j, 0j],
                [0j, (-1 + 0j), 0j, 0j],
                [0j, 0j, (1 + 0j), 0j],
                [0j, 0j, 0j, (-1 + 0j)],
            ],
        ],
        [
            [
                [0j, 0j, (1 + 0j), 0j],
                [0j, 0j, 0j, (1 + 0j)],
                [(1 + 0j), 0j, 0j, 0j],
                [0j, (1 + 0j), 0j, 0j],
            ],
            [
                [0j, 0j, 0j, (1 + 0j)],
                [0j, 0j, (1 + 0j), 0j],
                [0j, (1 + 0j), 0j, 0j],
                [(1 + 0j), 0j, 0j, 0j],
            ],
            [
                [0j, 0j, 0j, -1j],
                [0j, 0j, 1j, 0j],
                [0j, -1j, 0j, 0j],
                [1j, 0j, 0j, 0j],
            ],
            [
                [0j, 0j, (1 + 0j), 0j],
                [0j, 0j, 0j, (-1 + 0j)],
                [(1 + 0j), 0j, 0j, 0j],
                [0j, (-1 + 0j), 0j, 0j],
            ],
        ],
        [
            [
                [0j, 0j, -1j, 0j],
                [0j, 0j, 0j, -1j],
                [1j, 0j, 0j, 0j],
                [0j, 1j, 0j, 0j],
            ],
            [
                [0j, 0j, 0j, -1j],
                [0j, 0j, -1j, 0j],
                [0j, 1j, 0j, 0j],
                [1j, 0j, 0j, 0j],
            ],
            [
                [0j, 0j, 0j, (-1 + 0j)],
                [0j, 0j, (1 + 0j), 0j],
                [0j, (1 + 0j), 0j, 0j],
                [(-1 + 0j), 0j, 0j, 0j],
            ],
            [
                [0j, 0j, -1j, 0j],
                [0j, 0j, 0j, 1j],
                [1j, 0j, 0j, 0j],
                [0j, -1j, 0j, 0j],
            ],
        ],
        [
            [
                [(1 + 0j), 0j, 0j, 0j],
                [0j, (1 + 0j), 0j, 0j],
                [0j, 0j, (-1 + 0j), 0j],
                [0j, 0j, 0j, (-1 + 0j)],
            ],
            [
                [0j, (1 + 0j), 0j, 0j],
                [(1 + 0j), 0j, 0j, 0j],
                [0j, 0j, 0j, (-1 + 0j)],
                [0j, 0j, (-1 + 0j), 0j],
            ],
            [
                [0j, -1j, 0j, 0j],
                [1j, 0j, 0j, 0j],
                [0j, 0j, 0j, 1j],
                [0j, 0j, -1j, 0j],
            ],
            [
                [(1 + 0j), 0j, 0j, 0j],
                [0j, (-1 + 0j), 0j, 0j],
                [0j, 0j, (-1 + 0j), 0j],
                [0j, 0j, 0j, (1 + 0j)],
            ],
        ],
    ],
    dtype=np.complex128,
)


def spin_basis_to_z_basis(values: np.ndarray) -> np.ndarray:
    """values must be the factors in the spin basis sigma_alpha sigma_beta (alpha, beta in 0,x,y,z)"""
    return np.sum(
        density_matrix_builder_helper * values[:, :, np.newaxis, np.newaxis],
        axis=(0, 1),
    )


sigma_y_sigma_y = np.array(
    [
        [0.0 + 0.0j, 0.0 - 0.0j, 0.0 - 0.0j, -1.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, 1.0 - 0.0j, 0.0 - 0.0j],
        [0.0 + 0.0j, 1.0 - 0.0j, 0.0 + 0.0j, 0.0 - 0.0j],
        [-1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
    ]
)


def spin_flip_hermitian_matrix(matr: np.ndarray) -> np.ndarray:
    """sigma_y x sigma_y conj(matr) sigma_y x sigma_y"""
    conj = np.conjugate(matr)

    return sigma_y_sigma_y @ conj @ sigma_y_sigma_y


def concurrence_of_density_matrix(matr: np.ndarray) -> float:
    rho = matr

    rhoTilde = spin_flip_hermitian_matrix(matr)

    intermediate = rho @ rhoTilde

    eigenvals = np.flip(np.linalg.eigvals(intermediate))

    eigenvals = np.sort(np.real(eigenvals))[::-1]
    eigenvals[eigenvals < 0] = 0

    lambdas = np.sqrt(eigenvals)

    directConcurrence = np.max([0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]])

    return directConcurrence


def is_hermitian(test: np.ndarray, threshold: float) -> bool:
    return np.all(np.abs(test - np.matrix(test).H) < threshold)


def trace_is_one(test: np.ndarray, threshold: float) -> bool:
    trace = np.trace(test)
    return np.abs(trace - 1) < threshold


def get_reduced_density_matrix_in_z_basis_from_observations(
    spin_basis_measurements: np.ndarray, do_checks: bool = False, threshold=1e-4
) -> np.ndarray:
    spin_basis_measurements_real = np.real(spin_basis_measurements)
    imag_error = np.sum(np.abs(spin_basis_measurements_real - spin_basis_measurements))
    if imag_error > threshold:
        if do_checks:
            print(spin_basis_measurements)
            raise Exception("A complex part in the measurements was ommitted")
        else:
            print(
                f"Warning intermediate observables had imaginary part of {imag_error:.6f} that was omitted"
            )

    z_basis_values = spin_basis_to_z_basis(spin_basis_measurements / 4.0)
    if do_checks:
        if not is_hermitian(z_basis_values, threshold):
            print(z_basis_values)
            raise Exception("The density-matrix is not hermitian")
        if not trace_is_one(z_basis_values, threshold):
            print(z_basis_values)
            print(np.trace(z_basis_values))
            raise Exception("The trace of the density matrix is not one")

    return z_basis_values
