import numpy as np
def gram_shmidt_QR(A):
    m, n = A.shape
    R = np.zeros((n, n), dtype=A.dtype)
    Q = np.zeros((m, n), dtype=A.dtype)
    R[0][0] = np.linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0][0]
    for i in range(1, A.shape[1]):
        Q[:, i] = A[:, i]
        for j in range(0, i):
            R[j][i] = Q[:, j].T @ A[:, i]
            Q[:, i] = Q[:, i] - R[j][i] * Q[:, j]
        R[i][i] = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i][i]
    return Q, R

def modified_gram_schmidt_QR(A):
    m, n = A.shape
    R = np.zeros((n, n), dtype=A.dtype)
    Q = np.zeros((m, n), dtype=A.dtype)
    R[0][0] = np.linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0][0]
    for i in range(1, A.shape[1]):
        Q[:, i] = A[:, i]
        for j in range(0, i):
            R[j][i] = Q[:, j].T @ Q[:, i]
            Q[:, i] = Q[:, i] - R[j][i] * Q[:, j]
        R[i][i] = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i][i]
    return Q, R

def experiment(bits):
    DTYPE = np.float32 if bits == 32 else np.float64
    n = 50
    t = 2.0 ** np.arange(-8, 10, 2.0)
    t = t.astype(DTYPE)
    m = len(t)
    A = ( np.random.rand(n, m).astype(DTYPE) @ np.diag(t) @ np.random.rand(m, m).astype(DTYPE))
    Q_cgs, R_cgs = gram_shmidt_QR(A)
    Q_mgs, R_mgs = modified_gram_schmidt_QR(A)
    identity_m = np.eye(m, dtype=DTYPE)
    error_cgs = np.linalg.norm(identity_m - Q_cgs.T @ Q_cgs)
    error_mgs = np.linalg.norm(identity_m - Q_mgs.T @ Q_mgs)
    
    print(f"Classical Gram Schmidt Error: {error_cgs}")
    print(f"Modified Gram Schmidt Error: {error_mgs}")
    print(f"CGS Reconstruction Error: {np.linalg.norm(A - Q_cgs @ R_cgs)}")
    print(f"MGS Reconstruction Error: {np.linalg.norm(A - Q_mgs @ R_mgs)}")
    QTQ_cgs = Q_cgs.T @ Q_cgs
    QTQ_mgs = Q_mgs.T @ Q_mgs  

    I = np.eye(m, dtype = DTYPE)

    frob_cgs = frobenius(I - QTQ_cgs)
    frob_mgs = frobenius(I - QTQ_mgs)
    print(f"Frobenius norm for CGS: {frob_cgs}")
    print(f"Frobenius norm for MGS: {frob_mgs}")

def frobenius(A):
    sum_sq = 0
    m, n = A.shape
    for i in range(m):
        for j in range(n):
            sum_sq += A[i, j]**2
    return sum_sq**0.5 
print("Experiment with 32-bit:")
experiment(32)
print("\nExperiment with 64-bit:")
experiment(64)



 


