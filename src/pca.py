import numpy as np
from pathlib import Path
from PIL import Image
from dataset import load_data
from jacobi import jacobi

A_train, A_valid, y_train, y_valid = load_data()


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


# main flow
mean, F_train = average_face(A_train)

k = 30
eigenvalues, U, sweeps = eigenfaces(F_train, k)

Z_train = project(A_train, mean, U)  # weights Ω for all training faces
Z_valid = project(A_valid, mean, U)

y_pred = nearest_neighbor(Z_train, y_train, Z_valid)
accuracy = np.mean(y_pred == y_valid)

print("sweeps:", sweeps)
print("accuracy:", accuracy)

"""
def save_centered_faces(F, output_dir, shape=(64, 64)):
    output_dir.mkdir(exist_ok=True)

    for i, face in enumerate(F):
        image = face.reshape(shape)
        image = image - image.min()
        if image.max() != 0:
            image = image / image.max() * 255
        image = Image.fromarray(image.astype("uint8"))
        image.save(output_dir / f"centered_{i:03d}.png")


mean, F = average_face(A_train)
save_centered_faces(F, Path("centered_faces"))
"""
