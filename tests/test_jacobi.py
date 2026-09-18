import numpy as np
from jacobi import rotation_angle, rotate_matrix, off_diagonal, jacobi_cycle, jacobi


def test_off_diagonal():
    A = np.array([[1, 2], [3, 4]])
    # off-diagonal elements are 2 and 3 -> squares 4+9=13
    assert off_diagonal(A) == 13


def test_jacobi_2x2():
    A = np.array([[2, 1], [1, 2]])
    eigenvalues, eigenvectors = jacobi(A)
    expected = np.array([3, 1])

    assert np.allclose(eigenvalues, expected)


def test_diagonal_matrix():
    A = np.array([[5, 0, 0], [0, 9, 0], [0, 0, 4]])
    eigenvalues, eigenvectors = jacobi(A)

    assert np.allclose(eigenvalues, [9, 5, 4])


def test_eigenvectors():
    A = np.array([[5, 12, 7], [12, 9, 4], [7, 4, 4]])
    eigenvalues, eigenvectors = jacobi(A)

    assert np.allclose(A @ eigenvectors, eigenvalues * eigenvectors)  # Av=λv
