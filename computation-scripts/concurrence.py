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
    dtype=np.complex64,
)


def spin_basis_to_z_basis(values: np.ndarray) -> np.ndarray:
    """values must be the factors in the spin basis sigma_alpha sigma_beta (alpha, beta in 0,x,y,z)"""
    return np.sum(
        density_matrix_builder_helper * values[:, :, np.newaxis, np.newaxis],
        axis=(0, 1),
    )


measurement_to_spin_basis_conversion_helper = (
    np.array(
        [  # 1  a  c  e  f  g
            [1, 0, 0, 0, 0, 0],  # 00 == 1
            [1, 0, 0, 0, 0, 0],  # 0x == 1
            [-1j, 0, 2j, 0, 0, 0],  # 0y
            [1, 0, -2, 0, 0, 0],  # 0z
            [1, 0, 0, 0, 0, 0],  # x0 == 1
            [1, 0, 0, 0, 0, 0],  # xx == 1
            [-1j, 0, 0, 2j, 0, 2j],  # xy
            [1, 0, 0, -2, 0, -2],  # xz
            [-1j, 2j, 0, 0, 0, 0],  # y0
            [-1j, 0, 0, 2j, 2j, 0],  # yx
            [-1, 0, 0, 0, 2, 2],  # yy
            [-1j, -2j, 0, 2j, 4j, 2j],  # yz
            [1, -2, 0, 0, 0, 0],  # z0
            [1, 0, 0, -2, -2, 0],  # zx
            [-1j, 0, -2j, 2j, 2j, 4j],  # zy
            [3, 1, 1, -4, -4, -4],  # zz
        ],
        dtype=np.complex64,
    )
    / 4
)


def measurements_to_spin_basis(acefg: np.ndarray) -> np.ndarray:
    """acefg are the averaged measurements of observables:

    a: c_l,sigma (to left: (1-n_l,sigma) )

    c: c_m,sigma' (to left: (1-n_m,sigma') )

    e: c_l,sigma*c_m,sigma' (to left: ((1-n_l,sigma) * (1-n_m,sigma')) )

    f: c_l,sigma*c#_m,sigma' (to left: ((1-n_l,sigma) * n_m,sigma') )

    g: c#_l,sigma*c_m,sigma' (to left: (n_l,sigma * (1-n_m,sigma')) )
    """

    oacefg = np.insert(acefg, 0, 1)

    return (measurement_to_spin_basis_conversion_helper @ oacefg).reshape((4, 4))


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


def is_hermitian(test: np.ndarray) -> bool:
    return np.all(np.abs(test - np.matrix(test).H) < 1e-4)


def trace_is_one(test: np.ndarray) -> bool:
    purity = np.trace(test)
    return np.abs(purity - 1) < 1e-4


def calculate_concurrence(acefg: np.ndarray, do_checks: bool = False) -> float:
    """acefg are the averaged measurements of observables:

    a: c_l,sigma (to left: (1-n_l,sigma) )

    c: c_m,sigma' (to left: (1-n_m,sigma') )

    e: c_l,sigma*c_m,sigma' (to left: ((1-n_l,sigma) * (1-n_m,sigma')) )

    f: c_l,sigma*c#_m,sigma' (to left: ((1-n_l,sigma) * n_m,sigma') )

    g: c#_l,sigma*c_m,sigma' (to left: (n_l,sigma * (1-n_m,sigma')) )
    """
    acefg_real = np.real(acefg)
    imag_error = np.sum(np.abs(acefg_real - acefg))
    if imag_error > 1e-4:
        if do_checks:
            print(acefg)
            raise Exception("A complex part in the measurements was ommitted")
        else:
            print(
                f"Warning intermediate observables had imaginary part of {imag_error:.6f} that was omitted"
            )

    spin_basis_values = measurements_to_spin_basis(acefg_real)
    z_basis_values = spin_basis_to_z_basis(spin_basis_values)
    if do_checks:
        if not is_hermitian(z_basis_values):
            print(z_basis_values)
            raise Exception("The density-matrix is not hermitian")
        if not trace_is_one(z_basis_values):
            print(z_basis_values)
            raise Exception("The trace of the density matrix is not one")

    concurrence = concurrence_of_density_matrix(z_basis_values)

    return concurrence
