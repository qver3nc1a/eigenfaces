import numpy as np
import pytest
from jacobi import rotation_angle, rotate_matrix, off_diagonal, jacobi_cycle, jacobi


@pytest.mark.parametrize("scale", [1e-6, 1, 1e6, 1e8])
@pytest.mark.parametrize("seed", [40, 41, 42, 43, 44])
def test_sweeps(scale, seed):
    rng = np.random.default_rng(seed)
    B = rng.random((20, 20))
    A = scale * (B + np.transpose(B))

    eigenvalues, eigenvectors, sweeps = jacobi(A, tolerance=10 ** (-16), max_sweeps=50)
    assert sweeps < 15
    assert np.allclose(A @ eigenvectors, eigenvalues * eigenvectors)


def test_reject_non_symmetric():
    A = np.array([[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        jacobi(A)


def test_jacobi_input_unchanged():
    A = np.array([[1.0, 2.0], [2.0, 3.0]])
    orig_A = A.copy()
    jacobi(A)
    assert np.array_equal(A, orig_A)


def test_diagonal_not_rotated():
    A = np.diag([4.0, 8.0, 2.0])
    eigenvalues, eigenvectors, sweeps = jacobi(A)
    expected_eigenvalues = np.array([4.0, 8.0, 2.0])
    assert sweeps == 0


def test_zeros():
    A = np.zeros((5, 5))
    eigenvalues, eigenvectors, sweeps = jacobi(A)
    assert sweeps == 0
    assert np.array_equal(eigenvalues, np.zeros(5))
    assert np.allclose(np.transpose(eigenvectors) @ eigenvectors, np.eye(5))


def test_off_diagonal():
    A = np.array([[6.0, 4.0], [4.0, 8.0]])
    assert np.allclose(np.array([32.0]), off_diagonal(A))
