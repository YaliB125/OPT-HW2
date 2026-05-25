import numpy as np

A = np.array([
    [1, 2, 3, 4],
    [2, 4, -4, 8],
    [-5, 4, 1, 5],
    [5, 0, -3, -7]
])

### Part a
A_l1_norm = np.linalg.norm(A, ord=1)
A_lInfinity_norm = np.linalg.norm(A, ord=np.inf)

print("Part a:")
print(f"l1 norm for A is: {A_l1_norm}")
print(f"lInfinity norm for A is: {A_lInfinity_norm}")
print("==========================================")

### Part b
ATA = A.T @ A
ATA_eigenvalues, ATA_eigenvectors = np.linalg.eigh(ATA)
ATA_max_eigenvalue_idx = np.argmax(ATA_eigenvalues)
ATA_max_eigenvalue = ATA_eigenvalues[ATA_max_eigenvalue_idx]
A_l2_norm = np.sqrt(ATA_max_eigenvalue)

print("Part b:")
print(f"max eigenvalue for ATA is: {np.round(ATA_max_eigenvalue, 3)}")
print(f"the l2 norm for A is: {np.round(A_l2_norm, 3)}")
print("==========================================")

x_max = ATA_eigenvectors[ATA_max_eigenvalue_idx]

print(f"the vector that maximizes l2 norm for A is the eigenvector to the max eigenvalue of ATA: \n{np.round(x_max, 3)}")
print("==========================================")

### Part c
A_eigenvalues = np.linalg.eigvals(A)
A_spectral_radius = np.max(np.abs(A_eigenvalues))

print("Part c:")

if A_spectral_radius <= A_l1_norm + 1e-15: ### adding 1e-15 for roundoff errors
    print("l1 norm for A is higher that the spectral radius of A")

if A_spectral_radius <= A_lInfinity_norm + 1e-15: ### adding 1e-15 for roundoff errors
    print("lInfinity norm for A is higher that the spectral radius of A")

if A_spectral_radius <= A_l2_norm + 1e-15: ### adding 1e-15 for roundoff errors
    print("l2 norm for A is higher that the spectral radius of A")

