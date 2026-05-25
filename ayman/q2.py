import numpy as np

### Part a
def forward_substitution(L, b):
    n = L.shape[0]
    sol = np.zeros(n)

    for i in range(n):
        sum_i = b[i]

        for j in range(i):
            sum_i = sum_i - L[i][j] * sol[j]

        sol[i] = (1/L[i][i]) * sum_i
    
    return sol

def backward_substitution(U, b):
    n = U.shape[0]
    sol = np.zeros(n)

    for i in range(n-1, -1, -1):
        sum_i = b[i]

        for j in range(i+1, n):
            sum_i = sum_i - U[i][j] * sol[j]

        sol[i] = (1/U[i][i]) * sum_i
    
    return sol


A = np.array([
    [2, 1, 2],
    [1, -2, 1],
    [1, 2, 3],
    [1, 1, 1]
])
b = np.array([6, 1, 5, 2])

### Part b
ATA = A.T@A
L_colesky = np.linalg.cholesky(ATA)
y_colesky = forward_substitution(L_colesky, A.T @ b)
x_colesky = backward_substitution(L_colesky.T, y_colesky)

print("Part b")
print(f"the solution for the normal equations using the cholesky decomposition is: {np.round(x_colesky, 3)}")
print("==========================================")

### Part c
Q, R = np.linalg.qr(A, mode='reduced')
x_QR = backward_substitution(R, Q.T @ b)

print("Part C")
print(f"the solution for the normal equations using the QR decomposition is: {np.round(x_QR, 3)}")

def diagonal_solve(D, b):
    n = D.shape[0]
    sol = np.zeros(n)

    for i in range(n):
        sol[i] = (1/D[i][i]) * b[i] 
    
    return sol

U, S, V_T = np.linalg.svd(A, full_matrices=False)
y_SVD = diagonal_solve(np.diag(S), U.T @ b)
x_SVD = V_T.T @ y_SVD

print(f"the solution for the normal equations using the SVD decomposition is: {np.round(x_SVD, 3)}")

if np.allclose(x_colesky, x_QR, atol=1e-10) and np.allclose(x_QR, x_SVD, atol=1e-10):
    print(f"all decompositions returned the same solution: {np.round(x_colesky, 3)}")
