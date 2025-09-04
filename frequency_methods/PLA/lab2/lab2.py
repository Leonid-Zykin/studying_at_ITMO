import numpy as np
import matplotlib.pyplot as plt

def plot_transformation(matrix, title, draw_eigenvectors=False):
    polygon = np.array([
        [1, 0],
        [0, 1],
        [-1, -1],
        [1, 0]
    ]).T  # 2×N

    transformed = matrix @ polygon

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(*polygon, 'b-', label='Исходная фигура')
    ax.plot(*transformed, 'r--', label='Образ')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title(title)

    if draw_eigenvectors:
        eigvals, eigvecs = np.linalg.eig(matrix)
        for i in range(2):
            vec = eigvecs[:, i]
            ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
                      color='green', width=0.01)
            ax.text(vec[0], vec[1], f"$\\lambda={eigvals[i]:.2f}$", color='green')

    plt.show()

# Определяем все 16 матриц
M1 = np.array([[-3/5, 4/5], [4/5, 3/5]])
M2 = np.array([[1, 0], [3, 0]])
M3 = np.array([[np.cos(np.radians(40)), -np.sin(np.radians(40))],
               [np.sin(np.radians(40)), np.cos(np.radians(40))]])
M4 = np.array([[-1, 0], [0, -1]])
P = np.array([[np.cos(np.radians(50)), np.sin(np.radians(50))],
              [-np.sin(np.radians(50)), np.cos(np.radians(50))]])
M5 = P @ M1
M6 = np.array([[1, 1], [2, 3]])
M7 = np.array([[3, 1], [-2, 1]])
M8 = np.array([[1, 0], [5, -1]])
M9 = 2 * np.identity(2)
M10 = np.array([[5, 0], [0, 1]])
M11 = np.array([[1, 2], [2, -1]])
M12 = np.array([[1, 1], [0, 1]])
M13 = np.array([[0, -1], [1, 0]])
k = 2
M14 = k * np.identity(2)
A15 = np.array([[1, 1], [0, 1]])
B15 = np.array([[1, 0], [1, 1]])
A16 = np.array([[1, 2], [0, 1]])
B16 = np.array([[3, 0], [0, 3]])
AB15 = A15 @ B15
BA15 = B15 @ A15
AB16 = A16 @ B16
BA16 = B16 @ A16

# Список для визуализации
matrices = [
    ("M1", M1, True),
    ("M2", M2, False),
    ("M3", M3, False),
    ("M4", M4, False),
    ("M5", M5, False),
    ("M6", M6, False),
    ("M7", M7, False),
    ("M8", M8, False),
    ("M9", M9, False),
    ("M10", M10, False),
    ("M11", M11, True),
    ("M12", M12, True),
    ("M13", M13, False),
    ("M14", M14, True),
    ("A15", A15, True),
    ("B15", B15, True),
    ("AB15", AB15, True),
    ("BA15", BA15, True),
    ("A16", A16, True),
    ("B16", B16, True),
    ("AB16", AB16, True),
    ("BA16", BA16, True),
]

# Визуализация
for name, matrix, eig in matrices:
    plot_transformation(matrix, f"{name}", draw_eigenvectors=eig)
