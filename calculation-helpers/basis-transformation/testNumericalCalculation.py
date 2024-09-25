import numpy as np


def generateRandomHermitian4x4Matrix() -> np.ndarray:
    # Generate a random 4x4 complex matrix
    real_part = np.random.randn(4, 4)
    imag_part = np.random.randn(4, 4)

    # Create a complex matrix
    A = real_part + 1j * imag_part

    # Make the matrix Hermitian
    hermitian_matrix = (A + A.conj().T) / 2

    return hermitian_matrix


def sqrHermitianMatrixNumerically(matr: np.ndarray) -> np.ndarray:
    evalues, evectors = np.linalg.eigh(matr)
    sqrt_matrix = evectors * np.emath.sqrt(evalues) @ np.linalg.inv(evectors)
    return sqrt_matrix


def concurrenceOfDensityMatrix(mat: np.ndarray):
    rho = mat
    rhoTilde = np.conjugate(mat)

    sq = sqrHermitianMatrixNumerically(rho)

    RMatrix = sqrHermitianMatrixNumerically(sq @ rhoTilde @ sq)

    eigenvals = np.flip(np.linalg.eigvalsh(RMatrix))  # eigsh -> "in ascending order" !!

    print(eigenvals)

    directConcurrence = np.max(
        [0, eigenvals[0] - eigenvals[1] - eigenvals[2] - eigenvals[3]]
    )

    # comparison calculation
    largest = eigenvals[0]
    print(np.trace(RMatrix))
    print(2 * largest - np.trace(RMatrix))

    return directConcurrence


if __name__ == "__main__":

    test = generateRandomHermitian4x4Matrix()
    # print(test)

    sq = sqrHermitianMatrixNumerically(test)
    # print(sq)

    comp = sq @ sq - test
    # print((np.abs(np.real(comp))) < 1e-14)
    # print((np.abs(np.imag(comp))) < 1e-14)

    print(concurrenceOfDensityMatrix(test))
