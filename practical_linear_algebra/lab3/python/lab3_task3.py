import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

print("Задание 3: Перемещение кубика")

# Создание вершин кубика
vertices_cube = np.array([
    [-1, 1, 1, -1, -1, 1, 1, -1],
    [-1, -1, 1, 1, -1, -1, 1, 1],
    [-1, -1, -1, -1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
])

# Определение граней
faces_cube = np.array([
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [0, 1, 2, 3],
    [4, 5, 6, 7]
])

def create_translation_matrix(tx, ty, tz):
    """Создает матрицу перемещения"""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def create_scaling_matrix(sx, sy, sz):
    """Создает матрицу масштабирования"""
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

def apply_transformation(vertices, matrix):
    """Применяет матричное преобразование к вершинам"""
    return matrix @ vertices

def draw_shape(vertices, faces, color, alpha=0.7):
    """Отрисовывает 3D объект"""
    vertices_cartesian = (vertices[:3, :] / vertices[3, :]).T
    poly3d = Poly3DCollection(vertices_cartesian[faces],
                             facecolors=color,
                             edgecolors='black',
                             linewidths=0.5,
                             alpha=alpha)
    return poly3d

# Создание фигуры для демонстрации перемещений
fig = plt.figure(figsize=(15, 10))

# Различные перемещения
translations = [
    ([2, 0, 0], 'red', 'Перемещение по оси X'),
    ([0, 2, 0], 'green', 'Перемещение по оси Y'),
    ([0, 0, 1], 'blue', 'Перемещение по оси Z'),
    ([1, 1, 0], 'orange', 'Перемещение по диагонали XY'),
    ([1, 0, 1], 'purple', 'Перемещение по диагонали XZ'),
    ([0, 1, 1], 'brown', 'Перемещение по диагонали YZ')
]

for i, (translation, color, title) in enumerate(translations):
    ax = fig.add_subplot(2, 3, i+1, projection='3d', proj_type='ortho')
    
    # Исходный кубик
    poly3d_original = draw_shape(vertices_cube, faces_cube, 'lightgray', alpha=0.3)
    ax.add_collection3d(poly3d_original)
    
    # Перемещенный кубик
    T = create_translation_matrix(*translation)
    transformed_vertices = apply_transformation(vertices_cube, T)
    poly3d_transformed = draw_shape(transformed_vertices, faces_cube, color, alpha=0.7)
    ax.add_collection3d(poly3d_transformed)
    
    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 4)
    ax.set_zlim(-2, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(azim=-37.5, elev=30)

plt.tight_layout()
plt.savefig('images/task3/translation_transformations.png', dpi=300, bbox_inches='tight')
plt.close()

print("Демонстрация перемещений сохранена в images/task3/translation_transformations.png")

# Создание фигуры для сравнения TS и ST
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': '3d', 'proj_type': 'ortho'})

# Параметры для демонстрации
translation = [2, 1, 0]
scaling = [1.5, 0.8, 1.2]

# TS: сначала масштабирование, потом перемещение
S = create_scaling_matrix(*scaling)
T = create_translation_matrix(*translation)
TS_vertices = apply_transformation(apply_transformation(vertices_cube, S), T)

# ST: сначала перемещение, потом масштабирование
ST_vertices = apply_transformation(apply_transformation(vertices_cube, T), S)

# Отрисовка TS
poly3d_original = draw_shape(vertices_cube, faces_cube, 'lightgray', alpha=0.3)
ax1.add_collection3d(poly3d_original)

poly3d_ts = draw_shape(TS_vertices, faces_cube, 'red', alpha=0.7)
ax1.add_collection3d(poly3d_ts)

ax1.set_xlim(-2, 4)
ax1.set_ylim(-2, 4)
ax1.set_zlim(-2, 3)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('TS: Сначала масштабирование, потом перемещение')

# Отрисовка ST
poly3d_original = draw_shape(vertices_cube, faces_cube, 'lightgray', alpha=0.3)
ax2.add_collection3d(poly3d_original)

poly3d_st = draw_shape(ST_vertices, faces_cube, 'blue', alpha=0.7)
ax2.add_collection3d(poly3d_st)

ax2.set_xlim(-2, 4)
ax2.set_ylim(-2, 4)
ax2.set_zlim(-2, 3)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('ST: Сначала перемещение, потом масштабирование')

plt.tight_layout()
plt.savefig('images/task3/TS_vs_ST_translation.png', dpi=300, bbox_inches='tight')
plt.close()

print("Сравнение TS и ST для перемещения сохранено в images/task3/TS_vs_ST_translation.png")

print("Задание 3 выполнено!") 