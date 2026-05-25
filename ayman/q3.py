import numpy as np

### Part 1
def gram_schmidt_QR(A):
    m, n = A.shape

    Q = np.zeros((m, n), dtype=A.dtype)
    R = np.zeros((n, n), dtype=A.dtype)
    R[0][0] = np.linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0][0]

    for i in range(1, n):
        Q[:, i] = A[:, i]

        for j in range(i):
            R[j][i] = Q[:, j].T @ A[:, i]
            Q[:, i] = Q[:, i] - R[j][i] * Q[:, j]
        
        R[i][i] = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i][i]
    
    return Q, R

def modified_gram_schmidt_QR(A):
    m, n = A.shape

    Q = np.zeros((m, n), dtype=A.dtype)
    R = np.zeros((n, n), dtype=A.dtype)
    R[0][0] = np.linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0][0]

    for i in range(1, n):
        Q[:, i] = A[:, i]

        for j in range(i):
            R[j][i] = Q[:, j].T @ Q[:, i]
            Q[:, i] = Q[:, i] - R[j][i] * Q[:, j]
        
        R[i][i] = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i][i]
    
    return Q, R

def QR_difference(bits, GS_QR, MGS_QR):
    DTYPE = np.float32 if bits == 32 else np.float64

    n = 50
    t = 2.0 ** np.arange(-8, 10, 2.0)
    t = t.astype(DTYPE)
    m = len(t)

    A = (np.random.rand(n, m).astype(DTYPE) @ np.diag(t) @ np.random.rand(m, m).astype(DTYPE))

    Q_grame_schmidt, _ = GS_QR(A)
    Q_modified_grame_schmidt, _ = MGS_QR(A)

    I = np.eye(m, dtype = DTYPE)
    grame_schmidt_frobenius_norm = np.linalg.norm((Q_grame_schmidt.T @ Q_grame_schmidt) - I, 'fro')
    modified_grame_schmidt_frobenius_norm = np.linalg.norm((Q_modified_grame_schmidt.T @ Q_modified_grame_schmidt) - I, 'fro')

    print(f"grame schmidt QR decomposition Q.T @ Q - I frobenius norm is: {grame_schmidt_frobenius_norm}")
    print(f"modified grame schmidt QR decomposition Q.T @ Q - I frobenius norm is: {modified_grame_schmidt_frobenius_norm}")

    if grame_schmidt_frobenius_norm < modified_grame_schmidt_frobenius_norm:
        print("grame schmidt QR decomposition has a better decomposition")
    elif grame_schmidt_frobenius_norm > modified_grame_schmidt_frobenius_norm:
        print("modified grame schmidt QR decomposition has a better decomposition")
    else:
        print("both decomposition has the same result")

### Part c
print("Part c - compute decompositions with 32-bit:")
QR_difference(32, gram_schmidt_QR, modified_gram_schmidt_QR)
print("==========================================")

### Part d
print("Part d - compute decompositions with 64-bit:")
QR_difference(64, gram_schmidt_QR, modified_gram_schmidt_QR)
print("==========================================")

### Part e

def gram_schmidt_QR_print(A):
    m, n = A.shape

    Q = np.zeros((m, n), dtype=A.dtype)
    R = np.zeros((n, n), dtype=A.dtype)
    R[0][0] = np.linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0][0]

    for i in range(1, n):
        Q[:, i] = A[:, i]

        R_sequence = np.array([])

        for j in range(i):
            assert np.isclose(np.linalg.norm(Q[:, j]), 1, atol=1e-10), f"GS: the norm of q{j} is not 1"

            R[j][i] = Q[:, j].T @ A[:, i]
            Q[:, i] = Q[:, i] - R[j][i] * Q[:, j]

            R_sequence = np.append(R_sequence, R[j][i])
        
        print(f"GS R sequence {i} iteration: {R_sequence}")
        print("==========================================")
        
        R[i][i] = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i][i]
    
    return Q, R

def modified_gram_schmidt_QR_print(A):
    m, n = A.shape

    Q = np.zeros((m, n), dtype=A.dtype)
    R = np.zeros((n, n), dtype=A.dtype)
    R[0][0] = np.linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0][0]

    for i in range(1, n):
        Q[:, i] = A[:, i]

        R_sequence = np.array([])
        qi_norm_sequence = np.array([np.linalg.norm(Q[:, i])])

        for j in range(i):
            assert np.isclose(np.linalg.norm(Q[:, j]), 1, atol=1e-10), f"MGS: the norm of q{j} is not 1"

            R[j][i] = Q[:, j].T @ Q[:, i]
            Q[:, i] = Q[:, i] - R[j][i] * Q[:, j]

            new_qi_norm = np.linalg.norm(Q[:, i])

            assert 1e-10 + qi_norm_sequence[-1] >= new_qi_norm, f"MSG: the norm of qi is not monotonically non increasing"

            R_sequence = np.append(R_sequence, R[j][i])
            qi_norm_sequence = np.append(qi_norm_sequence, new_qi_norm)
        
        print(f"MGS R sequence {i} iteration: {R_sequence}")
        print(f"MGS qi norm sequence {i} iteration: {qi_norm_sequence}")
        print("==========================================")

        R[i][i] = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i][i]

    return Q, R

print("Part e - compute decompositions with 32-bit and check R sequence and qi norm sequence:")
QR_difference(32, gram_schmidt_QR_print, modified_gram_schmidt_QR_print)
