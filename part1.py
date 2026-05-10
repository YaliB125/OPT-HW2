import numpy as np
A = np.array([
    [1, 2, 3, 4],
    [2, 4, -4, 8],
    [-5, 4, 1, 5],
    [5, 0, -3, -7]
])

ATA = A.T @ A
eigenvalues, eigenvectors = np.linalg.eigh(ATA)
print("Eigenvalues of A^T A:", eigenvalues)
singular_values = np.sqrt(eigenvalues)
print("Singular values of A:", singular_values)
idx = np.argmax(eigenvalues)
x_max = eigenvectors[:, idx]
print("x_max:", x_max)

eigvals_A = np.linalg.eigvals(A)
print("Eigenvalues of A:", eigvals_A)
spectral_radius = np.max(np.abs(eigvals_A))
print("Spectral radius of A:", spectral_radius)