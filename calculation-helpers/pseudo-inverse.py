import numpy as np

n = 8
random_matrix = np.random.rand(n, n)

print("Random Matrix:")
print(random_matrix)

inverse_matrix = np.linalg.inv(random_matrix)

print("\nInverse Matrix:")
print(inverse_matrix)

pseudo_inverse_matrix = np.linalg.pinv(random_matrix)

print("\nPseudo Inverse Matrix:")
print(pseudo_inverse_matrix)

print("\nPseudo/non difference:")
print(pseudo_inverse_matrix - inverse_matrix)

product = np.dot(inverse_matrix, random_matrix)

print("\nProduct:")
print(product)

pseudo_product = np.dot(pseudo_inverse_matrix, random_matrix)

print("\nPseudo Product:")
print(pseudo_product)
