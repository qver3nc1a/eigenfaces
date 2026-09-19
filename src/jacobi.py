# input: square symmetrical numpy array N * N
# output: array of N eigenvalues (desc) and N * N matrix of eigenvectors (in columns)
from math import atan2, sin, cos
import numpy as np


def rotation_angle(A, p, q):
    # calculate the rotation angle (tan(2O) = 2a_pq/a_qq-a_pp)
    return 0.5 * atan2(2 * A[p][q], (A[q][q] - A[p][p]))


def rotate_matrix(A, V, p, q, O):
    # build the rotation matrix R (and return new A with A_pq = 0)
    R = np.identity(len(A[0]))
    R[p][p] = cos(O)
    R[q][q] = cos(O)
    R[p][q] = sin(O)
    R[q][p] = -sin(O)

    return np.transpose(R) @ A @ R, V @ R


def off_diagonal(A):
    # return sum of off-diagonal squares to know when to stop
    A_off = A.copy()
    np.fill_diagonal(A_off, 0)

    return np.sum(A_off * A_off)


def jacobi_cycle(A, V):
    n = A.shape[0]

    for p in range(n - 1):
        for q in range(p + 1, n):
            O = rotation_angle(A, p, q)
            A, V = rotate_matrix(A, V, p, q, O)
    return A, V


def jacobi(A, tolerance=10 ** (-10), max_sweeps=50):
    # run jacobi
    if not np.array_equal(A, np.transpose(A)):
        return  # error

    sweeps = 0
    A = A.copy()  # keep global matrix unchanged
    V = np.identity(A.shape[0])

    while True:
        curr_off_sq_sum = off_diagonal(A)

        if curr_off_sq_sum < tolerance:
            break
        if sweeps >= max_sweeps:
            break

        A, V = jacobi_cycle(A, V)
        sweeps += 1

    eigenvalues = np.diag(A)
    eigenvectors = V

    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    return eigenvalues, eigenvectors, sweeps


rng = np.random.default_rng(seed=42)

B1 = rng.random((20, 20))
A1 = 1e6 * (B1 + np.transpose(B1))

B2 = rng.random((20, 20))
A2 = 1e6 * (B2 + np.transpose(B2))

B3 = rng.random((20, 20))
A3 = 1e6 * (B3 + np.transpose(B3))

B4 = rng.random((20, 20))
A4 = 1e6 * (B4 + np.transpose(B4))

B5 = rng.random((20, 20))
A5 = 1e6 * (B5 + np.transpose(B5))

e1, e2, sweeps1 = jacobi(A1, tolerance=10 ** (-10), max_sweeps=50)
e1, e2, sweeps2 = jacobi(A2, tolerance=10 ** (-10), max_sweeps=50)
e1, e2, sweeps3 = jacobi(A3, tolerance=10 ** (-10), max_sweeps=50)
e1, e2, sweeps4 = jacobi(A4, tolerance=10 ** (-10), max_sweeps=50)
e1, e2, sweeps5 = jacobi(A5, tolerance=10 ** (-10), max_sweeps=50)

print(sweeps1, sweeps2, sweeps3, sweeps4, sweeps5)
