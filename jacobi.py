# input: square symmetrical numpy array N * N
# output: array of N eigenvalues (desc) and N * N matrix of eigenvectors (in columns)
from math import atan2, sin, cos
import numpy as np


def rotation_angle(A, p, q):
    # calculate the rotation angle (tan(2O) = 2a_pq/a_qq-a_pp)
    p -= 1
    q -= 1
    return 0.5 * atan2(2 * A[p][q], (A[q][q] - A[p][p]))


def rotate_matrix(A, p, q, O):
    # build the rotation matrix R (and return new A with A_pq = 0)
    p -= 1
    q -= 1
    R = np.identity(len(A[0]))
    R[p][p] = cos(O)
    R[q][q] = cos(O)
    R[p][q] = sin(O)
    R[q][p] = -sin(O)

    return np.transpose(R) @ A @ R


def off_diagonal(A):
    # return sum of off-diagonal squares to know when to stop
    sq_A = A * A
    sq_sum = np.sum(sq_A)
    sq_diag = np.sum(np.diag(sq_A))

    off_sq_sum = sq_sum - sq_diag
    return off_sq_sum


def jacobi(A, tolerance=10 ** (-10), max_sweeps=50):
    pass
    # run jacobi


A = np.array([[4, 2, 7], [2, 5, 3], [7, 3, 8]])
print(off_diagonal(A))
O = rotation_angle(A, 1, 3)
print(O)
print(rotate_matrix(A, 1, 3, O))
