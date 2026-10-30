import numpy as np
from dataset import load_data
from jacobi import jacobi
from pca import average_face, covariance, eigenfaces, project, nearest_neighbor

A_train, A_valid, y_train, y_valid = load_data()

# (280, 4096) = 40 people * 7 (training) images = 280, 64x64 pixels = 4096
print("Training data shape: ", A_train.shape)

# (120, 4096) = 40 people * 3 (validation) images = 120, 64x64 pixels = 4096
print("Validation data shape: ", A_valid.shape)

print("Running PCA pipeline...")
mean, F_train = average_face(A_train)

# optimal k to explain 95% of variance
use_n_percent_k = False
# change percentage to explain different variance
percentage = 0.95
if use_n_percent_k:
    L = covariance(F_train)
    eigenvalues, V, sweeps = jacobi(L)

    total_eig = sum(eigenvalues)
    curr = 0
    for i, value in enumerate(eigenvalues):
        curr += value
        if curr / total_eig >= percentage:
            k = i + 1
            break

    V_k = V[:, :k]

    U = np.transpose(F_train) @ V_k
    U = U / np.linalg.norm(U, axis=0)

# change to see different results
else:
    k = 80
    eigenvalues, U, sweeps = eigenfaces(F_train, k)

Z_train = project(A_train, mean, U)
Z_valid = project(A_valid, mean, U)

predictions = nearest_neighbor(Z_train, y_train, Z_valid)

correct = np.sum(predictions == y_valid)
total = len(y_valid)
accuracy = correct / total

print("Eigenfaces used:", U.shape[1])
print("Jacobi sweeps:", sweeps)
print("Correct predictions:", correct)
print("Total validation images:", total)
if use_n_percent_k:
    print("Used k explains", percentage * 100, " percent of variance")
else:
    print("Used k = ", k)
print("Accuracy:", f"{accuracy:.2%}")
