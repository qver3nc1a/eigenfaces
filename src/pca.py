import numpy as np

# import load data function from dataset when ready :)
A_train, l_train, A_valid, l_valid = load_data()


def average_face(A):
    mean = np.mean(A, axis=0)  # across rows so across faces
    F = A - mean
    return mean, F
