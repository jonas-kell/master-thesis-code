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
    ]
)


def combineValues(values) -> np.array:
    return np.sum(
        densityMatrixBuilderArray * values[:, :, np.newaxis, np.newaxis], axis=(0, 1)
    )


testValues = np.array(
    [
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

print(combineValues(testValues))
