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
