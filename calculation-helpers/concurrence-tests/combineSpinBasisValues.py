import numpy as np

# from "spinToZBasisTransformation.py"
densityMatrixBuilderArray = np.array(
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


def combineValues(values: np.ndarray) -> np.ndarray:
    """values must be the factors in the spin basis sigma_alpha sigma_beta (alpha, beta in 0,x,y,z)"""
    return np.sum(
        densityMatrixBuilderArray * values[:, :, np.newaxis, np.newaxis], axis=(0, 1)
    )


def isHermitian(test):
    print(np.abs(test - np.matrix(test).H) < 1e-4)


def printPurity(test):
    purity = np.trace(test @ test)
    print(purity)
    print("pure" if np.abs(purity - 1) < 1e-4 else "not pure")


if __name__ == "__main__":
    testValues = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    print(combineValues(testValues))
    isHermitian(combineValues(testValues))
    printPurity(combineValues(testValues))
