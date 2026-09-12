# input: square symmetrical numpy array N * N
# output: array of N eigenvalues (desc) and N * N matrix of eigenvectors (in columns)
from math import atan, sin, cos
import numpy as np


def rotation_angle(A, p, q):
    # calculates the rotation angle (tan(2O) = 2a_pq/a_qq-a_pp)
    return 0.5 * atan(2 * A[p][q] / (A[q][q] - A[p][p]))


def rotate_matrix(A, p, q, O):
    # build the rotation matrix R (and return new A with A_pq = 0)
    R = np.identity(len(A[0]))
    R[p][p] = cos(O)
    R[q][q] = cos(O)
    R[p][q] = -sin(O)
    R[q][p] = sin(O)

    return np.transpose(R) @ A @ R
