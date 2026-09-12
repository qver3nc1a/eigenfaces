# input: square symmetrical numpy array N * N
# output: array of N eigenvalues (desc) and N * N matrix of eigenvectors (in columns)
from math import atan2, sin, cos
import numpy as np


def rotation_angle(A, p, q):
    # calculate the rotation angle (tan(2O) = 2a_pq/a_qq-a_pp)
    return 0.5 * atan2(2 * A[p - 1][q - 1], (A[q - 1][q - 1] - A[p - 1][p - 1]))


def rotate_matrix(A, p, q, O):
    # build the rotation matrix R (and return new A with A_pq = 0)
    R = np.identity(len(A[0]))
    R[p - 1][p - 1] = cos(O)
    R[q - 1][q - 1] = cos(O)
    R[p - 1][q - 1] = -sin(O)
    R[q - 1][p - 1] = sin(O)

    return np.transpose(R) @ A @ R


A = np.array([[4, 2, 7], [2, 5, 3], [7, 3, 8]])
O = rotation_angle(A, 1, 3)
print(O)
print(rotate_matrix(A, 1, 3, O))


def off_diagonal(A):
    pass
    # return sum of off-diagonal norms to know when to stop


def jacobi(A, tolerance=10 ** (-10), max_sweeps=50):
    pass
    # run jacobi
