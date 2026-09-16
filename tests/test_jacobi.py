import numpy as np
from jacobi import rotation_angle, rotate_matrix, off_diagonal, jacobi_cycle, jacobi


def test_off_diagonal():
    A = np.array([[1, 2], [3, 4]])
    # off-diagonal elements are 2 and 3 -> squares 4+9=13
    assert off_diagonal(A) == 13
