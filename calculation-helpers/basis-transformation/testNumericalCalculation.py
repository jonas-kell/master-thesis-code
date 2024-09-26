import numpy as np
from typing import Tuple


def generateRandomHermitian4x4Matrix() -> np.ndarray:
    # Generate a random 4x4 complex matrix
    real_part = 2 * np.random.rand(4, 4) - 1
    imag_part = 2 * np.random.rand(4, 4) - 1

    # Create a complex matrix
    A = real_part + 1j * imag_part

    # Make the matrix Hermitian
    hermitian_matrix = (A + A.conj().T) / 2

    return hermitian_matrix


# -> (matrix, factors)
# order: |uu>, |ud>, |du>, |dd>
def generateRandomTwoSpinPureDensityMatrix() -> Tuple[np.ndarray, np.ndarray]:
    real_part = 2 * np.random.rand(4) - 1
    imag_part = 2 * np.random.rand(4) - 1

    coefficients = real_part + 1j * imag_part

    normalization = np.sum(np.abs(coefficients) ** 2)
    normalized_coefficients = coefficients / normalization

    alpha = normalized_coefficients[0]
    beta = normalized_coefficients[1]
    gamma = normalized_coefficients[2]
    delta = normalized_coefficients[3]

    densityMatrix = np.array(
        [
            [
                alpha * np.conjugate(alpha),
                alpha * np.conjugate(beta),
                alpha * np.conjugate(gamma),
                alpha * np.conjugate(delta),
            ],
            [
                beta * np.conjugate(alpha),
                beta * np.conjugate(beta),
                beta * np.conjugate(gamma),
                beta * np.conjugate(delta),
            ],
            [
                gamma * np.conjugate(alpha),
                gamma * np.conjugate(beta),
                gamma * np.conjugate(gamma),
                gamma * np.conjugate(delta),
            ],
            [
                delta * np.conjugate(alpha),
                delta * np.conjugate(beta),
                delta * np.conjugate(gamma),
                delta * np.conjugate(delta),
            ],
        ]
    )

    return (densityMatrix, coefficients)


def sqrHermitianMatrixNumerically(matr: np.ndarray) -> np.ndarray:
    evalues, evectors = np.linalg.eigh(matr)
    sqrt_matrix = evectors * np.emath.sqrt(evalues) @ np.linalg.inv(evectors)
    return sqrt_matrix


def spinFlipHermitianMatrix(matr: np.ndarray) -> np.ndarray:
    # Spin flip operation result
    # ⎡a  b  c  d⎤
    # ⎢          ⎥
    # ⎢_         ⎥
    # ⎢b  e  f  g⎥
    # ⎢          ⎥
    # ⎢_  _      ⎥
    # ⎢c  f  h  k⎥
    # ⎢          ⎥
    # ⎢_  _  _   ⎥
    # ⎣d  g  k  l⎦
    # ----->
    # ⎡l   -k  -g  d ⎤
    # ⎢              ⎥
    # ⎢ _            ⎥
    # ⎢-k  h   f   -c⎥
    # ⎢              ⎥
    # ⎢ _  _         ⎥
    # ⎢-g  f   e   -b⎥
    # ⎢              ⎥
    # ⎢_    _   _    ⎥
    # ⎣d   -c  -b  a ⎦

    antiTranspose = matr[::-1, ::-1].T
    mask = np.array(
        [
            [1, -1, -1, 1],
            [-1, 1, 1, -1],
            [-1, 1, 1, -1],
            [1, -1, -1, 1],
        ]
    )

    return mask * antiTranspose


def spinFlipHermitianMatrixSpin(matr: np.ndarray) -> np.ndarray:
    # sigma_y x sigma_y conj(matr) sigma_y x sigma_y

    sigma_y_sigma_y = np.array(
        [
            [0.0 + 0.0j, 0.0 - 0.0j, 0.0 - 0.0j, -1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 - 0.0j, 0.0 - 0.0j],
            [0.0 + 0.0j, 1.0 - 0.0j, 0.0 + 0.0j, 0.0 - 0.0j],
            [-1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ]
    )

    conj = np.conjugate(matr)

    return sigma_y_sigma_y @ conj @ sigma_y_sigma_y


def concurrenceOfDensityMatrix(mat: np.ndarray, useSpin: bool):
    rho = mat

    if useSpin:
        rhoTilde = spinFlipHermitianMatrixSpin(mat)
    else:
        rhoTilde = spinFlipHermitianMatrix(mat)

    sqrtMat = sqrHermitianMatrixNumerically(rho)

    RMatrix = sqrHermitianMatrixNumerically(sqrtMat @ rhoTilde @ sqrtMat)

    eigenvals = np.flip(np.linalg.eigvalsh(RMatrix))  # eigsh -> "in ascending order" !!

    directConcurrence = np.max(
        [0, eigenvals[0] - eigenvals[1] - eigenvals[2] - eigenvals[3]]
    )

    # comparison calculation
    # !! caution, I think that the complex part here is rather large...
    # print("Trace/eigenval check")
    # print(np.sum(eigenvals) - np.trace(RMatrix))
    # largest = eigenvals[0]
    # print(2 * largest - np.trace(RMatrix))

    return directConcurrence


def concurrencePerPureStatesComponentsPaper(alpha, beta, gamma, delta):
    transitionMatrix = (
        np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [1j, 0.0, 0.0, -1j],
                [0.0, 1j, 1j, 0.0],
                [0.0, 1.0, -1.0, 0.0],
            ]
        )
        / 4.0
    )

    coefficients = transitionMatrix @ np.array([alpha, beta, gamma, delta]).T

    return np.abs(np.sum(coefficients**2))


def concurrencePerPureStatesComponentsPaperZBasis(alpha, beta, gamma, delta):
    coefficients = np.array([alpha, beta, gamma, delta])

    return np.abs(np.sum(coefficients**2))


def concurrencePerPureStatesComponentsZBasis(alpha, beta, gamma, delta):
    return 2 * np.abs(alpha * delta - beta * gamma)


def concurrencePerPureStatesComponents(alpha, beta, gamma, delta):
    transitionMatrix = (
        np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [1j, 0.0, 0.0, -1j],
                [0.0, 1j, 1j, 0.0],
                [0.0, 1.0, -1.0, 0.0],
            ]
        )
        / 4.0
    )

    alphaMagicBasis, betaMagicBasis, gammaMagicBasis, deltaMagicBasis = (
        transitionMatrix @ np.array([alpha, beta, gamma, delta]).T
    )

    return 2 * np.abs(
        alphaMagicBasis * deltaMagicBasis - betaMagicBasis * gammaMagicBasis
    )


def nonSymmConcurrenceOfDensityMatrix(mat: np.ndarray, useSpin: bool):
    rho = mat

    if useSpin:
        rhoTilde = spinFlipHermitianMatrixSpin(mat)
    else:
        rhoTilde = spinFlipHermitianMatrix(mat)

    intermediate = rho @ rhoTilde

    eigenvals = np.flip(np.linalg.eigvals(intermediate))

    # print(eigenvals)
    eigenvals = np.sort(np.real(eigenvals))[::-1]
    # print(eigenvals)

    directConcurrence = np.max(
        [0, eigenvals[0] - eigenvals[1] - eigenvals[2] - eigenvals[3]]
    )

    return directConcurrence


if __name__ == "__main__":

    # test = generateRandomHermitian4x4Matrix()
    test, factors = generateRandomTwoSpinPureDensityMatrix()
    print("Test density matrix")
    print(test)
    print()

    print("Test factors")
    print(factors)
    print()

    # ! sqrt of matrix check
    # sq = sqrHermitianMatrixNumerically(test)
    # print(sq)
    # comp = sq @ sq - test
    # print((np.abs(np.real(comp))) < 1e-14)
    # print((np.abs(np.imag(comp))) < 1e-14)

    print("With spin flip matrix")
    print(concurrenceOfDensityMatrix(test, True))
    print()

    print("With basis transformation")
    print(concurrenceOfDensityMatrix(test, False))
    print()

    print("Assym With spin flip matrix")
    print(nonSymmConcurrenceOfDensityMatrix(test, True))
    print()

    print("Assym basis transformation")
    print(nonSymmConcurrenceOfDensityMatrix(test, False))
    print()

    alphaFactor, betaFactor, gammaFactor, deltaFactor = factors
    print("Direct calculation as per paper Magic-Basis")
    print(
        concurrencePerPureStatesComponentsPaper(
            alphaFactor, betaFactor, gammaFactor, deltaFactor
        )
    )
    print()
    print("Direct calculation as per paper Z-Basis")
    print(
        concurrencePerPureStatesComponentsPaperZBasis(
            alphaFactor, betaFactor, gammaFactor, deltaFactor
        )
    )
    print()
    print("Direct calculation as per wiki Magic-Basis")
    print(
        concurrencePerPureStatesComponents(
            alphaFactor, betaFactor, gammaFactor, deltaFactor
        )
    )
    print()
    print("Direct calculation as per wiki Z-Basis")
    print(
        concurrencePerPureStatesComponentsZBasis(
            alphaFactor, betaFactor, gammaFactor, deltaFactor
        )
    )
    print()
