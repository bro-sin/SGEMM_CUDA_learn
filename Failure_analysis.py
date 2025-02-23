import numpy as np
from matplotlib import pyplot as plt
import seaborn


def load_matrix_from_file(file_path, dim: int):
    A = np.loadtxt(
        file_path,
        dtype=np.float32,
        delimiter=",",
        skiprows=1,
        max_rows=dim,
    )

    B = np.loadtxt(
        file_path,
        dtype=np.float32,
        delimiter=",",
        skiprows=1 + dim + 1,
        max_rows=dim,
    )
    C = np.loadtxt(
        file_path,
        dtype=np.float32,
        delimiter=",",
        skiprows=1 + dim + 1 + dim + 1,
        max_rows=dim,
    )
    Should = np.loadtxt(
        file_path,
        dtype=np.float32,
        delimiter=",",
        skiprows=1 + dim + 1 + dim + 1 + dim + 1,
        max_rows=dim,
    )

    return A, B, C, Should


def main():
    file_path = "build/matrixValidationFailure.txt"
    matrices = load_matrix_from_file(file_path, 64)

    for i, matrix in enumerate(matrices):
        print(f"Matrix {i + 1}:")
        print(matrix.shape)
    C = matrices[2]
    Should = matrices[3]
    diff = np.power(C - Should,2)
    seaborn.heatmap(diff)
    plt.show()


if __name__ == "__main__":
    main()
