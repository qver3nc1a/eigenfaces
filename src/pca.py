import numpy as np
from dataset import load_data
from jacobi import jacobi


# calculate average face Ψ=1/M*∑​Γ_i and Φi​=Γi​−Ψ​
def average_face(A):
    mean = np.mean(A, axis=0)  # across rows so across faces
    F = A - mean
    return mean, F


# covariance matrix C
def covariance(F):
    return F @ np.transpose(F)


# jacobi for eigenvalues and eigenvectors -> eigenfaces (multiply by F^T)
def eigenfaces(F, k):
    L = covariance(F)
    eigenvalues, V, sweeps = jacobi(L)

    # choose first k
    V_k = V[:, :k]  # first k columns
    U = np.transpose(F) @ V_k

    # normalize
    U = U / np.linalg.norm(U, axis=0)

    return eigenvalues[:k], U, sweeps


# weights (Γ - Ψ) @ U
def project(A, mean, U):
    return (A - mean) @ U


# distances |Ωi​−Ω|^2
def nearest_neighbor(Z_train, y_train, Z_valid):
    predictions = []

    for z in Z_valid:
        distances = np.sum((Z_train - z) ** 2, axis=1)
        predictions.append(y_train[np.argmin(distances)])  # closest known

    return np.array(predictions)
