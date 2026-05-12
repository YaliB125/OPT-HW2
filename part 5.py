import numpy as np
A= np.array([[5, 4, 0],
    [4, 5, 0],
    [0, 0, 2],
    [0, 0, 1]])
B= np.array([[15, 14, 0],
    [14, 15, 0],
    [0, 0, 2],
    [0, 0, 1]])
UA, sA, VTA = np.linalg.svd(A, full_matrices=False)
UB, sB, VTB = np.linalg.svd(B, full_matrices=False)

def best_rank_1(U, s, VT, shape):
    m, n = shape
    sigma1 = s[0]      
    u1 = U[:, 0]       
    v1t = VT[0, :]     
    
    X = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            X[i][j] = sigma1 * u1[i] * v1t[j]
    return X

def frobenius(A):
    sum_sq = 0
    m, n = A.shape
    for i in range(m):
        for j in range(n):
            sum_sq += A[i, j]**2
    return sum_sq**0.5 

A1 = best_rank_1(UA, sA, VTA, A.shape)
B1 = best_rank_1(UB, sB, VTB, B.shape)

error_A = frobenius(A - A1)
rel_error_A = error_A / frobenius(A)
error_B = frobenius(B - B1)
rel_error_B = error_B / frobenius(B)

print("Best rank-1 approximation A1:\n", A1)
print("Frobenius norm of A - A1:", error_A)
print("Relative error for A:", rel_error_A)
print("\nBest rank-1 approximation B1:\n", B1)
print("Frobenius norm of B - B1:", error_B)
print("Relative error for B:", rel_error_B)