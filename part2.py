import numpy as np
def forward_substitution(L, b):
    X=np.zeros(L.shape[0])
    for i in range(L.shape[0]):
        sum_val = b[i]
        for j in range(i):
            sum_val -= L[i][j] * X[j]
        X[i] = sum_val / L[i][i]
    return X
 
def backward_substitution(U, b):
    X=np.zeros(U.shape[0])
    for i in range(U.shape[0]-1, -1, -1):
        sum_val = b[i]
        for j in range(i+1, U.shape[0]):
            sum_val -= U[i][j] * X[j]
        X[i] = sum_val / U[i][i]
    return X

def algo_cholesky(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        sum1 = 0
        for k in range (i):
            sum1 += L[i][k] ** 2
        L[i][i] = (A[i][i] - sum1) ** 0.5
        for j in range(i+1, n):
            sum2 =0
            for k in range(i):
                sum2 += L[i][k] * L[j][k]
            L[j][i] = (A[j][i] - sum2) / L[i][i]
    return L

A = np.array([
    [2, 1, 2],
    [1, -2, 1],
    [1, 2, 3],
    [1, 1, 1]
])
b = np.array([6, 1, 5, 2])

print ("solution using Cholesky:")
L = algo_cholesky(A.T @ A)
y_colesky = forward_substitution(L, A.T @ b)
x_colesky = backward_substitution(L.T, y_colesky)
print("Solution x:", x_colesky)

print("\nSolution using QR:")
Q, R = np.linalg.qr(A)
y_qr = Q.T @ b
x_qr = backward_substitution(R, y_qr)
print("Solution x:", x_qr)

print("\nSolution using SVD:")
U, SIG, V = np.linalg.svd(A, full_matrices=False)
UTb = U.T @ b
y_svd = UTb / SIG       
x_svd = V.T @ y_svd
print("Solution x:", x_svd)


