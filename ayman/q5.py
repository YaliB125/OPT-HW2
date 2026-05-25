import numpy as np

A= np.array(
    [
        [5, 4, 0],
        [4, 5, 0],
        [0, 0, 2],
        [0, 0, 1]
    ]
)

B= np.array(
    [
        [15, 14, 0],
        [14, 15, 0],
        [0, 0, 2],
        [0, 0, 1]
    ]
)

### Part a
UA, SA, VTA = np.linalg.svd(A, full_matrices=False)
UB, SB, VTB = np.linalg.svd(B, full_matrices=False)

def best_rank1_approximation(siagm1, u1, v1T):
    return siagm1 * np.outer(u1, v1T)

A1 = best_rank1_approximation(SA[0], UA[:, 0], VTA[0, :])
B1 = best_rank1_approximation(SB[0], UB[:, 0], VTB[0, :])

print("Part a")
print(f"best rank1 approximation for A is: \n{np.round(A1, 3)}")
print(f"best rank1 approximation for B is: \n{np.round(B1, 3)}")
print("==========================================")

### Part b
approximation_error_A = np.linalg.norm(A - A1, 'fro')
approximation_error_B = np.linalg.norm(B - B1, 'fro')

print("Part b")
print(f"approximation error for A is: {approximation_error_A}")
print(f"approximation error for B is: {approximation_error_B}")
print("==========================================")

relative_error_A = approximation_error_A / np.linalg.norm(A, 'fro')
relative_error_B = approximation_error_B / np.linalg.norm(B, 'fro')

print(f"relative error for A is: {relative_error_A}")
print(f"relative error for B is: {relative_error_B}")
print("==========================================")
