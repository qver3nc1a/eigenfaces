import numpy as np
import pytest
from jacobi import rotation_angle, rotate_matrix, off_diagonal, jacobi_cycle, jacobi


def test_off_diagonal():
    A = np.array([[1, 2], [3, 4]])
    # off-diagonal elements are 2 and 3 -> squares 4+9=13
    assert off_diagonal(A) == 13


def test_off_diagonal1e8():
    A = np.diag([1e8, 2e8, 3e8])
    A[0, 1] = 1e-3
    A[1, 0] = 1e-3

    off = off_diagonal(A)
    assert off == 2 * 1e-6


def test_jacobi_2x2():
    A = np.array([[2, 1], [1, 2]])
    eigenvalues, eigenvectors = jacobi(A)[:2]
    expected = np.array([3, 1])

    assert np.allclose(eigenvalues, expected)


def test_diagonal_matrix():
    A = np.array([[5, 0, 0], [0, 9, 0], [0, 0, 4]])
    eigenvalues, eigenvectors = jacobi(A)[:2]

    assert np.allclose(eigenvalues, [9, 5, 4])


def test_eigenvectors():
    A = np.array([[5, 12, 7], [12, 9, 4], [7, 4, 4]])
    eigenvalues, eigenvectors = jacobi(A)[:2]

    assert np.allclose(A @ eigenvectors, eigenvalues * eigenvectors)  # Av=λv


@pytest.mark.parametrize("scale", [1e-6, 1, 1e6, 1e8])
@pytest.mark.parametrize("seed", [40, 41, 42, 43, 44])
def test_sweeps(scale, seed):
    rng = np.random.default_rng(seed)
    B = rng.random((20, 20))
    A = scale * (B + np.transpose(B))

    e1, e2, sweeps = jacobi(A, tolerance=10 ** (-10), max_sweeps=50)
    assert sweeps <= 50
