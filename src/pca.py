import numpy as np
from dataset import load_data
from jacobi import jacobi


def average_face(A):
    """Calculate average face (Ψ=1/M*∑​Γ_i) and centered face matrix (Φi​=Γi​−Ψ​)

    Args:
        A: matrix containing face data, 1 face per row

    Returns:
        mean, F: a tuple containing the average face and the centered face matrix
    """
    mean = np.mean(A, axis=0)  # across rows so across faces
    F = A - mean
    return mean, F


def covariance(F):
    """Compute the covariance matrix from centered face data

    Args:
        F: centered face matrix

    Returns:
        covariance matrix
    """
    return F @ np.transpose(F)


def eigenfaces(F, k):
    """Compute the first k eigenfaces from centered face data

    Args:
        F: Centered face matrix
        k: Number of eigenfaces to keep

    Returns:
        eigenvalues[:k], U, sweeps: A tuple containing the selected k eigenvalues, eigenfaces,
        and the number of Jacobi sweeps
    """
    L = covariance(F)
    eigenvalues, V, sweeps = jacobi(L)

    # choose first k
    V_k = V[:, :k]  # first k columns
    U = np.transpose(F) @ V_k

    # normalize
    U = U / np.linalg.norm(U, axis=0)

    return eigenvalues[:k], U, sweeps


def project(A, mean, U):
    """Project face data into eigenface coordinates

    Args:
        A: Matrix containing face data, 1 face per row
        mean: Average face
        U: Matrix with eigenfaces in columns

    Returns:
        Matrix of projection weights ((Γ - Ψ) @ U) for each face
    """
    return (A - mean) @ U


# distances |Ωi​−Ω|^2
def nearest_neighbor(Z_train, y_train, Z_valid):
    """Predict labels for validation faces

    Args:
        Z_train: Training faces (represented by projection weights)
        y_train: Labels of the training faces
        Z_valid: Validation faces (represented by projection weights)

    Returns:
        NumPy array containing one predicted label per validation face
    """
    predictions = []

    for z in Z_valid:
        distances = np.sum((Z_train - z) ** 2, axis=1)
        predictions.append(y_train[np.argmin(distances)])  # closest known

    return np.array(predictions)
