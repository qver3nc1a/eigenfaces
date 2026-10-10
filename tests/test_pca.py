import numpy as np
import pytest
from pca import average_face, covariance, eigenfaces, project, nearest_neighbor


def test_average_face():
    A = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]])
    mean, F = average_face(A)
    expected_mean, expected_F = (
        np.array([4.4, 5.5, 6.6]),
        np.array(
            [
                [-3.3000000e00, -3.3000000e00, -3.3000000e00],
                [8.8817842e-16, 0.0000000e00, 8.8817842e-16],
                [3.3000000e00, 3.3000000e00, 3.3000000e00],
            ]
        ),
    )
    assert np.allclose(mean, expected_mean)
    assert np.allclose(F, expected_F)


def test_covariance_values():
    F = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    C = covariance(F)

    expected = np.array(
        [
            [5.0, 11.0],
            [11.0, 25.0],
        ]
    )

    assert np.allclose(C, expected)


def test_covariance_symmetric():
    F = np.array(
        [
            [-3.3000000e00, -3.3000000e00, -3.3000000e00],
            [8.8817842e-16, 0.0000000e00, 8.8817842e-16],
            [3.3000000e00, 3.3000000e00, 3.3000000e00],
        ]
    )
    C = covariance(F)
    assert C.shape == (3, 3)
    assert np.allclose(C, np.transpose(C))


def test_eigenfaces_sanity():
    F = np.array([[1.0, 2.0], [2.0, -1.0], [0.0, 1.0], [-2.0, 1.0]])
    eigenvalues, U, sweeps = eigenfaces(F, k=1)
    assert eigenvalues.shape == (1,)
    assert U.shape == (2, 1)
    assert sweeps > 0
    assert np.allclose(np.linalg.norm(U), 1)


def test_project():
    A = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    mean = np.array([1.0, 4.0])
    U = np.array([1.0, 0.0])

    proj = project(A, mean, U)

    expected_proj = np.array([[0.0, 2.0]])
    assert np.allclose(proj, expected_proj)


def test_nearest_neighbor():
    Z_train = np.array([[0.0, 0.0], [20.0, 0.0]])
    y_train = np.array(["A", "B"])

    Z_valid = np.array([[0.1, 0.3], [15.0, 22.0]])

    preds = nearest_neighbor(Z_train, y_train, Z_valid)
    assert np.array_equal(preds, np.array(["A", "B"]))


def test_pca():
    A_train = np.array(
        [[0.0, 0.0], [0.1, 0.2], [-0.1, 0.0], [20.0, 20.0], [21.0, 19.0], [19.9, 20.1]]
    )
    y_train = np.array(["A", "A", "A", "B", "B", "B"])
    A_valid = np.array([[0.01, 0.0], [19.9, 20.2]])
    y_valid = np.array(["A", "B"])

    mean, F = average_face(A_train)
    eigenvalues, U, sweeps = eigenfaces(F, k=1)

    Z_train = project(A_train, mean, U)
    Z_valid = project(A_valid, mean, U)
    y_pred = nearest_neighbor(Z_train, y_train, Z_valid)

    assert np.array_equal(y_pred, y_valid)
