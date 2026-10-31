import numpy as np
from PIL import Image
from pathlib import Path
from dataset import load_data
from jacobi import jacobi
from pca import average_face, covariance, eigenfaces, project, nearest_neighbor


def save_centered_faces(F, labels, output_dir, shape=(64, 64)):
    if output_dir.exists():
        for old_face in output_dir.rglob("*.png"):
            old_face.unlink()
    output_dir.mkdir(exist_ok=True)

    for i, face in enumerate(F):
        person = labels[i]
        folder = output_dir / f"s{person:02d}"
        folder.mkdir(exist_ok=True)
        image = face.reshape(shape)
        image = image - image.min()
        if image.max() != 0:
            image = image / image.max() * 255

        image = Image.fromarray(image.astype("uint8"))
        image.save(folder / f"{i:02d}.png")


A_train, A_valid, y_train, y_valid = load_data()

# (280, 4096) = 40 people * 7 (training) images = 280, 64x64 pixels = 4096
print("Training data shape: ", A_train.shape)

# (120, 4096) = 40 people * 3 (validation) images = 120, 64x64 pixels = 4096
print("Validation data shape: ", A_valid.shape)

mean, F_train = average_face(A_train)

# optimal k to explain a percentage of variance
print(
    "Would you like to use an optimal number of eigenvalues k to explain a fixed percentage of variance? T/F"
)
while True:
    answer = input()
    if answer == "T" or answer == "t" or answer == "True" or answer == "true":
        use_n_percent_k = True
        break
    elif answer == "F" or answer == "f" or answer == "False" or answer == "false":
        use_n_percent_k = False
    else:
        print("Please choose either T (True) or F (False)")

# change percentage to explain different variance
if use_n_percent_k:
    print(
        "What percentage of variance would you like the chosen eigenvalues to explain? Please give your answer as a positive float (less than 1). Example: 0.95"
    )
    while True:
        var = float(input())
        if var >= 0 and var <= 1:
            percentage = var
            break
        else:
            print(
                "Please give your answer as a positive float that is less than or equal to 1. Example: 0.95"
            )
    print("Running PCA pipeline...")

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
    print("Choose k (1 ≤ k ≤ 280)")
    while True:
        ch_k = int(input())
        if ch_k >= 1 and ch_k <= 280:
            k = ch_k
            break
        else:
            print("Please make sure k is an integer and (1 ≤ k ≤ 280)")
    print("Running PCA pipeline...")
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
    print("Used k explains", percentage * 100, "percent of variance")
else:
    print("Used k = ", k)
print("Accuracy:", f"{accuracy:.2%}")

print("Would you like to view the centered faces? T/F")
while True:
    answer = input()
    if answer == "T" or answer == "t" or answer == "True" or answer == "true":
        centered = True
        break
    elif answer == "F" or answer == "f" or answer == "False" or answer == "false":
        use_n_percent_k = False
    else:
        print("Please choose either T (True) or F (False)")

if centered:
    print("Saving centered faces...")
    save_centered_faces(F_train, y_train, Path("centered_faces"))
    print("You can view the centered faces in centered_faces/")
