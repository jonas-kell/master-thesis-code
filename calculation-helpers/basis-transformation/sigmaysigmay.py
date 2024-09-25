import numpy as np

sigmay = np.array([[0, -1j], [1j, 0]])

sigmaysigmay = np.array(
    [
        [sigmay[row // 2][col // 2] * sigmay[row % 2][col % 2] for col in range(4)]
        for row in range(4)
    ]
)

print(sigmay)
print(sigmaysigmay)

print()
print()

test = np.array(
    [
        [1, 2 + 2j, 3 + 3j, 4 + 4j],
        [5 + 5j, 6, 7 + 7j, 8 + 8j],
        [9 + 9j, 10 + 10j, 11, 12 + 12j],
        [13 + 13j, 14 + 14j, 15 + 15j, 16],
    ]
)

print(test)
print(sigmaysigmay @ np.conjugate(test) @ sigmaysigmay)
