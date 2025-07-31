import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

# Создание вершин кубика (однородные координаты)
vertices_cube = np.array([
    [-1, 1, 1, -1, -1, 1, 1, -1],    # x координаты
    [-1, -1, 1, 1, -1, -1, 1, 1],    # y координаты  
    [-1, -1, -1, -1, 1, 1, 1, 1],    # z координаты
    [1, 1, 1, 1, 1, 1, 1, 1]         # однородная координата w
])

# Определение граней кубика
faces_cube = np.array([
    [0, 1, 5, 4],  # передняя грань
    [1, 2, 6, 5],  # правая грань
    [2, 3, 7, 6],  # задняя грань
    [3, 0, 4, 7],  # левая грань
    [0, 1, 2, 3],  # нижняя грань
    [4, 5, 6, 7]   # верхняя грань
])

def create_scaling_matrix(sx, sy, sz):
    """
    Создание матрицы масштабирования
    sx, sy, sz - коэффициенты масштабирования по осям X, Y, Z
    """
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

def apply_transformation(vertices, matrix):
    """
    Применение матрицы преобразования к вершинам
    """
    return matrix @ vertices

def draw_shape(vertices, faces, color, alpha=0.7):
    """
    Функция для отрисовки 3D объекта
    """
    vertices_cartesian = (vertices[:3, :] / vertices[3, :]).T
    poly3d = Poly3DCollection(vertices_cartesian[faces], 
                             facecolors=color, 
                             edgecolors='black', 
                             linewidths=0.5,
                             alpha=alpha)
    return poly3d

# Создание различных матриц масштабирования
scaling_matrices = {
    'uniform': create_scaling_matrix(2, 2, 2),      # Равномерное масштабирование
    'non_uniform': create_scaling_matrix(1.5, 2, 0.8),  # Неравномерное масштабирование
    'stretch_x': create_scaling_matrix(3, 1, 1),    # Растяжение по X
    'stretch_y': create_scaling_matrix(1, 3, 1),    # Растяжение по Y
    'stretch_z': create_scaling_matrix(1, 1, 3),    # Растяжение по Z
    'squash': create_scaling_matrix(0.5, 0.5, 2)   # Сжатие по X,Y, растяжение по Z
}

# Создание фигуры с подграфиками
fig = plt.figure(figsize=(20, 12))

for i, (name, matrix) in enumerate(scaling_matrices.items(), 1):
    ax = fig.add_subplot(2, 3, i, projection='3d', proj_type='ortho')
    
    # Применение преобразования
    transformed_vertices = apply_transformation(vertices_cube, matrix)
    
    # Отрисовка исходного кубика (прозрачный)
    poly3d_original = draw_shape(vertices_cube, faces_cube, 'lightblue', alpha=0.3)
    ax.add_collection3d(poly3d_original)
    
    # Отрисовка преобразованного кубика
    poly3d_transformed = draw_shape(transformed_vertices, faces_cube, 'red', alpha=0.7)
    ax.add_collection3d(poly3d_transformed)
    
    # Настройка осей
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    
    # Настройка меток
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Масштабирование: {name}\nМатрица: [{matrix[0,0]:.1f}, {matrix[1,1]:.1f}, {matrix[2,2]:.1f}]')
    
    # Настройка угла обзора
    ax.view_init(azim=-37.5, elev=30)

plt.tight_layout()
plt.savefig('images/task2/scaling_transformations.png', dpi=300, bbox_inches='tight')
plt.close()

print("Масштабирование кубика выполнено и сохранено в images/task2/scaling_transformations.png")

# Дополнительная визуализация: сравнение TS и ST
fig = plt.figure(figsize=(15, 6))

# Создание матриц масштабирования и перемещения
S = create_scaling_matrix(2, 1.5, 0.8)
T = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
])

# Применение TS (сначала масштабирование, потом перемещение)
vertices_TS = apply_transformation(apply_transformation(vertices_cube, S), T)

# Применение ST (сначала перемещение, потом масштабирование)
vertices_ST = apply_transformation(apply_transformation(vertices_cube, T), S)

# Отрисовка результатов
ax1 = fig.add_subplot(1, 2, 1, projection='3d', proj_type='ortho')
poly3d_TS = draw_shape(vertices_TS, faces_cube, 'green', alpha=0.7)
ax1.add_collection3d(poly3d_TS)
ax1.set_title('TS (масштабирование → перемещение)')
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_zlim(-3, 3)
ax1.view_init(azim=-37.5, elev=30)

ax2 = fig.add_subplot(1, 2, 2, projection='3d', proj_type='ortho')
poly3d_ST = draw_shape(vertices_ST, faces_cube, 'orange', alpha=0.7)
ax2.add_collection3d(poly3d_ST)
ax2.set_title('ST (перемещение → масштабирование)')
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_zlim(-3, 3)
ax2.view_init(azim=-37.5, elev=30)

plt.tight_layout()
plt.savefig('images/task2/TS_vs_ST_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Сравнение TS и ST выполнено и сохранено в images/task2/TS_vs_ST_comparison.png") 