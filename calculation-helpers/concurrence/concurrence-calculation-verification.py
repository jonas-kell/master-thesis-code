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

    normalization = np.sqrt(np.sum(np.abs(coefficients) ** 2))
    normalized_coefficients = coefficients / normalization

    alpha = normalized_coefficients[0]
    beta = normalized_coefficients[1]
    gamma = normalized_coefficients[2]
    delta = normalized_coefficients[3]

    checkFactors(normalized_coefficients)
    checkFactors(np.array([alpha, beta, gamma, delta]))

    densityMatrix = buildDensityMatrixFromCoefficients(alpha, beta, gamma, delta)

    return (densityMatrix, normalized_coefficients)


def checkFactors(fact: np.ndarray):
    if np.abs(np.sum(np.abs(fact) ** 2) - 1.0) > 1e-6:
        raise Exception("Not normalized", np.sum(np.abs(fact) ** 2))


def buildDensityMatrixFromCoefficients(alpha, beta, gamma, delta):
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

    if np.abs(np.trace(densityMatrix) - 1.0) > 1e-6:
        raise Exception("Density Matrix not of trace 1")

    return densityMatrix


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


transitionMatrix = np.array(
    [
        [1.0, 0.0, 0.0, 1.0],
        [1j, 0.0, 0.0, -1j],
        [0.0, 1j, 1j, 0.0],
        [0.0, 1.0, -1.0, 0.0],
    ]
) / np.sqrt(2)


def concurrencePerPureStatesComponentsPaper(alpha, beta, gamma, delta):
    checkFactors(np.array([alpha, beta, gamma, delta]))
    coefficients = transitionMatrix @ np.array([alpha, beta, gamma, delta]).T
    checkFactors(coefficients)

    return np.abs(np.sum(coefficients**2))


def concurrencePerPureStatesComponents(alpha, beta, gamma, delta):
    alphaMagicBasis, betaMagicBasis, gammaMagicBasis, deltaMagicBasis = (
        transitionMatrix @ np.array([alpha, beta, gamma, delta]).T
    )
    checkFactors(
        np.array([alphaMagicBasis, betaMagicBasis, gammaMagicBasis, deltaMagicBasis])
    )

    return 2 * np.abs(
        alphaMagicBasis * deltaMagicBasis - betaMagicBasis * gammaMagicBasis
    )


def concurrencePerPureStatesComponentsPaperZBasis(alpha, beta, gamma, delta):
    coefficients = np.array([alpha, beta, gamma, delta])
    checkFactors(coefficients)

    return np.abs(np.sum(coefficients**2))


def concurrencePerPureStatesComponentsZBasis(alpha, beta, gamma, delta):
    checkFactors(np.array([alpha, beta, gamma, delta]))

    return 2 * np.abs(alpha * delta - beta * gamma)


def nonSymmConcurrenceOfDensityMatrix(mat: np.ndarray, useSpin: bool):
    rho = mat

    if useSpin:
        rhoTilde = spinFlipHermitianMatrixSpin(mat)
    else:
        rhoTilde = spinFlipHermitianMatrix(mat)

    intermediate = rho @ rhoTilde

    eigenvals = np.flip(np.linalg.eigvals(intermediate))

    eigenvals = np.sort(np.real(eigenvals))[::-1]
    eigenvals[eigenvals < 0] = 0

    lambdas = np.sqrt(eigenvals)

    directConcurrence = np.max([0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]])

    return directConcurrence


if __name__ == "__main__":

    # ! sqrt of matrix check
    # hermitianMt = generateRandomHermitian4x4Matrix()
    # sq = sqrHermitianMatrixNumerically(hermitianMt)
    # print(hermitianMt)
    # comp = sq @ sq - hermitianMt
    # print((np.abs(np.real(comp))) < 1e-14)
    # print((np.abs(np.imag(comp))) < 1e-14)

    testDM, factors = generateRandomTwoSpinPureDensityMatrix()
    # factors = np.array([1, 0, 0, 0])  # / np.sqrt(2)

    # ! Matrix, extracted out of the running simulation
    # setting numerical-time-evolution to start_time: float = 6.2
    # testDM = np.array(
    #     [
    #         [
    #             0.22218453 + 0.0j,
    #             -0.20357603 + 0.00696778j,
    #             0.22605906 + 0.01585645j,
    #             -0.22150816 - 0.00412117j,
    #         ],
    #         [
    #             -0.20357603 - 0.00696778j,
    #             0.19813227 + 0.0j,
    #             -0.18833556 - 0.01967613j,
    #             0.20341119 + 0.01075346j,
    #         ],
    #         [
    #             0.22605906 - 0.01585645j,
    #             -0.18833556 + 0.01967613j,
    #             0.35749866 + 0.0j,
    #             -0.2263149 + 0.0116486j,
    #         ],
    #         [
    #             -0.22150816 + 0.00412117j,
    #             0.20341119 - 0.01075346j,
    #             -0.2263149 - 0.0116486j,
    #             0.22218453 + 0.0j,
    #         ],
    #     ]
    # )
    # extraction from the numerical time evolution through diagonalization
    # testDM = np.array(
    #     [
    #         [
    #             0.25 + 0.0j,
    #             -0.18959892 + 0.04716978j,
    #             0.21783418 - 0.01935684j,
    #             -0.24008819 + 0.06813023j,
    #         ],
    #         [
    #             -0.18959892 - 0.04716978j,
    #             0.17218176 + 0.0j,
    #             -0.15196817 - 0.01717849j,
    #             0.19527422 - 0.00638107j,
    #         ],
    #         [
    #             0.21783418 + 0.01935684j,
    #             -0.15196817 + 0.01717849j,
    #             0.32781824 + 0.0j,
    #             -0.21484426 + 0.0408456j,
    #         ],
    #         [
    #             -0.24008819 - 0.06813023j,
    #             0.19527422 + 0.00638107j,
    #             -0.21484426 - 0.0408456j,
    #             0.25 + 0.0j,
    #         ],
    #     ]
    # )

    # ! perform checks on the properties of the matrix

    if np.abs(np.trace(testDM) - 1) > 1e-6:
        raise Exception("Density matrix not trace 1")

    if not np.all(np.abs((testDM - np.conj(testDM.T))) < 1e-6):
        raise Exception("Density matrix not hermitian")

    print("Test density matrix")
    print(testDM)
    print()

    print("Test factors")
    print(factors)
    print()

    # ! calculate concurrence

    print("With spin flip matrix")
    print(concurrenceOfDensityMatrix(testDM, True))
    print()

    print("With basis transformation")
    print(concurrenceOfDensityMatrix(testDM, False))
    print()

    print("Assym With spin flip matrix")
    print(nonSymmConcurrenceOfDensityMatrix(testDM, True))
    print()

    print("Assym basis transformation")
    print(nonSymmConcurrenceOfDensityMatrix(testDM, False))
    print()

    alphaFactor, betaFactor, gammaFactor, deltaFactor = factors
    print("Direct calculation as per paper Magic-Basis")
    print(
        concurrencePerPureStatesComponentsPaper(
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
    print("Direct calculation as per paper Z-Basis (different!)")
    print(
        concurrencePerPureStatesComponentsPaperZBasis(
            alphaFactor, betaFactor, gammaFactor, deltaFactor
        )
    )
    print()
    print("Direct calculation as per wiki Magic-Basis (different!)")
    print(
        concurrencePerPureStatesComponents(
            alphaFactor, betaFactor, gammaFactor, deltaFactor
        )
    )
    print()
