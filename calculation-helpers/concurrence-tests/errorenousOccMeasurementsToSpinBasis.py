import numpy as np
from combineSpinBasisValues import combineValues, isHermitian, printPurity


def assemble_coefficients(a, c):
    """The factors a,b,c,d stand for c_l c#_l c_m c#_m"""

    b = 1 - a
    d = 1 - c

    return np.array([a, b, c, d, a * c, a * d, b * c, b * d])


def matrix_obvious(a, b, c, d, e, f, g, h) -> np.ndarray:

    return (
        np.array(
            [
                [1, c + d, 1j * (c - d), 2 * d - 1],
                [a + b, e + f + g + h, 1j * (e + g - f - h), 2 * (f + h) - a - b],
                [
                    1j * (a - b),
                    1j * (e + f - g - h),
                    f - h - e + g,
                    1j * (2 * (f - h) - a + b),
                ],
                [
                    2 * b - 1,
                    2 * (g + h) - c - d,
                    1j * (2 * (g - h) - c + d),
                    4 * h - b - d + 1,
                ],
            ],
            dtype=np.complex128,
        )
        / 4
    )


factor_to_spin_basis_conversion_helper = (
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
        dtype=np.complex128,
    )
    / 4
)


def compile_in_spin_basis_form_averaged(acefg: np.ndarray) -> np.ndarray:

    oacefg = np.insert(acefg, 0, 1)

    return (factor_to_spin_basis_conversion_helper @ oacefg).reshape((4, 4))


if __name__ == "__main__":
    coeff = np.zeros(8)
    sample = 30
    for _ in range(sample):
        av, cv = np.random.rand(2)
        coeff += assemble_coefficients(av, cv)
    coeff = coeff / sample

    # TEST WITH SPECIFIC COEFFICIENTS
    coeff = assemble_coefficients(1, 0)

    # split for obvious implementation
    av, bv, cv, dv, ev, fv, gv, hv = coeff

    print(coeff)
    spin_basis_factors = compile_in_spin_basis_form_averaged(
        np.array([av, cv, ev, fv, gv])
    )
    print(spin_basis_factors)
    print(
        np.abs(matrix_obvious(av, bv, cv, dv, ev, fv, gv, hv) - spin_basis_factors)
        < 1e-4
    )

    # This is not necissarily hermitian, as the inputs are not real observables (missing scaling by the e^(E-E) that makes the imaginary parts vanish)
    print("Hermiticity and purity check")
    z_basis_mat = combineValues(spin_basis_factors)
    print(z_basis_mat)
    isHermitian(z_basis_mat)
    printPurity(z_basis_mat)
