# input: square symmetrical numpy array N * N
# output: array of N eigenvalues (desc) and N * N matrix of eigenvectors (in columns)
from math import atan2, sin, cos
import numpy as np


def rotation_angle(A, p, q):
    # calculate the rotation angle (tan(2O) = 2a_pq/a_qq-a_pp)
    return 0.5 * atan2(2 * A[p][q], (A[q][q] - A[p][p]))


def rotate_matrix(A, p, q, O):
    # build the rotation matrix R (and return new A with A_pq = 0)
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


def jacobi_cycle(A):
    # max off-diagonal value norm
    mask = np.ones(A.shape)
    np.fill_diagonal(mask, 0)
    max_off_value = np.abs(A[mask]).max()

    # coordinates
    p, q = np.where(np.abs(A) == max_off_value)
    p, q = p[0], q[0]

    # rotation angle
    O = rotation_angle(A, p, q)

    # R and A_new
    A_new = rotate_matrix(A, p, q, O)
    return A_new


def jacobi(A, tolerance=10 ** (-10), max_sweeps=50):
    # run jacobi
    if not np.array_equal(A, np.transpose(A)):
        return  # error
    sweeps = 0
    while True:
        curr_off_sq_sum = off_diagonal(A)
        if curr_off_sq_sum < tolerance:
            break
        if sweeps >= max_sweeps:
            break

        A = jacobi_cycle(A)
        sweeps += 1
    return A


A = np.array([[4, 2, 7], [2, 5, 3], [7, 3, 8]])
print(off_diagonal(A))
O = rotation_angle(A, 1, 3)
print(O)
print(rotate_matrix(A, 1, 3, O))
