import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
from scipy.linalg import expm

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

def create_rotation_matrix_around_axis(v, theta):
    """
    Создание матрицы вращения вокруг произвольной оси
    v - вектор оси вращения [vx, vy, vz]
    theta - угол вращения в радианах
    """
    # Нормализация вектора
    v_norm = np.array(v) / np.linalg.norm(v)
    vx, vy, vz = v_norm
    
    # Создание матрицы J согласно формуле из задания
    J = np.array([
        [0, -vz, vy, 0],
        [vz, 0, -vx, 0],
        [-vy, vx, 0, 0],
        [0, 0, 0, 0]
    ])
    
    # Вычисление матрицы вращения Rv(θ) = e^(Jθ)
    R = expm(J * theta)
    
    # Установка последнего элемента в 1 для однородных координат
    R[3, 3] = 1
    
    return R

def create_translation_matrix(tx, ty, tz):
    """
    Создание матрицы перемещения
    """
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def apply_transformation(vertices, matrix):
    """Применение матрицы преобразования к вершинам"""
    return matrix @ vertices

def draw_shape(vertices, faces, color, alpha=0.7):
    """Функция для отрисовки 3D объекта"""
    vertices_cartesian = (vertices[:3, :] / vertices[3, :]).T
    poly3d = Poly3DCollection(vertices_cartesian[faces], 
                             facecolors=color, 
                             edgecolors='black', 
                             linewidths=0.5,
                             alpha=alpha)
    return poly3d

def rotate_around_vertex(vertices, vertex_index, rotation_axis, angle):
    """
    Вращение кубика вокруг заданной вершины
    vertices - вершины кубика
    vertex_index - индекс вершины, вокруг которой вращаем
    rotation_axis - ось вращения
    angle - угол вращения в радианах
    """
    # Получаем координаты вершины вращения
    pivot_vertex = vertices[:3, vertex_index] / vertices[3, vertex_index]
    
    # Создаем матрицы преобразования
    T1 = create_translation_matrix(-pivot_vertex[0], -pivot_vertex[1], -pivot_vertex[2])  # Перенос в начало координат
    R = create_rotation_matrix_around_axis(rotation_axis, angle)  # Вращение
    T2 = create_translation_matrix(pivot_vertex[0], pivot_vertex[1], pivot_vertex[2])  # Обратный перенос
    
    # Композиция преобразований: T2 * R * T1
    transformation_matrix = T2 @ R @ T1
    
    return apply_transformation(vertices, transformation_matrix)

# Создание фигуры для демонстрации вращений вокруг вершин
fig = plt.figure(figsize=(20, 15))

# Определение вершин кубика для вращения
vertex_positions = {
    0: "(-1, -1, -1)",  # Нижняя левая передняя
    1: "(1, -1, -1)",   # Нижняя правая передняя
    2: "(1, 1, -1)",    # Нижняя правая задняя
    3: "(-1, 1, -1)",   # Нижняя левая задняя
    4: "(-1, -1, 1)",   # Верхняя левая передняя
    5: "(1, -1, 1)",    # Верхняя правая передняя
    6: "(1, 1, 1)",     # Верхняя правая задняя
    7: "(-1, 1, 1)"     # Верхняя левая задняя
}

# Вращения вокруг разных вершин
rotations = [
    (0, [1, 1, 1], np.pi/3, "Вращение вокруг вершины 0 (-1,-1,-1)\nна угол 60° вокруг оси [1,1,1]"),
    (1, [0, 1, 0], np.pi/2, "Вращение вокруг вершины 1 (1,-1,-1)\nна угол 90° вокруг оси Y"),
    (2, [1, 0, 0], np.pi/4, "Вращение вокруг вершины 2 (1,1,-1)\nна угол 45° вокруг оси X"),
    (3, [0, 0, 1], np.pi/6, "Вращение вокруг вершины 3 (-1,1,-1)\nна угол 30° вокруг оси Z"),
    (4, [1, 1, 0], np.pi/2, "Вращение вокруг вершины 4 (-1,-1,1)\nна угол 90° вокруг оси [1,1,0]"),
    (5, [1, 0, 1], np.pi/3, "Вращение вокруг вершины 5 (1,-1,1)\nна угол 60° вокруг оси [1,0,1]")
]

for i, (vertex_idx, axis, angle, title) in enumerate(rotations, 1):
    ax = fig.add_subplot(2, 3, i, projection='3d', proj_type='ortho')
    
    # Исходный кубик
    poly3d_original = draw_shape(vertices_cube, faces_cube, 'lightblue', alpha=0.3)
    ax.add_collection3d(poly3d_original)
    
    # Вращенный кубик
    vertices_rotated = rotate_around_vertex(vertices_cube, vertex_idx, axis, angle)
    poly3d_rotated = draw_shape(vertices_rotated, faces_cube, 'red', alpha=0.7)
    ax.add_collection3d(poly3d_rotated)
    
    # Отрисовка оси вращения
    pivot_vertex = vertices_cube[:3, vertex_idx] / vertices_cube[3, vertex_idx]
    axis_norm = np.array(axis) / np.linalg.norm(axis)
    start = pivot_vertex - 1.5 * axis_norm
    end = pivot_vertex + 1.5 * axis_norm
    
    ax.quiver(start[0], start[1], start[2], 
              end[0] - start[0], end[1] - start[1], end[2] - start[2],
              color='red', arrow_length_ratio=0.2, linewidth=3)
    
    # Отрисовка точки вращения
    ax.scatter([pivot_vertex[0]], [pivot_vertex[1]], [pivot_vertex[2]], 
               color='red', s=100, marker='o')
    
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.view_init(azim=-37.5, elev=30)

plt.tight_layout()
plt.savefig('images/task5/rotation_around_vertices.png', dpi=300, bbox_inches='tight')
plt.close()

print("Вращения вокруг вершин выполнены и сохранены в images/task5/rotation_around_vertices.png")

# Дополнительная визуализация: сравнение вращений вокруг разных вершин
fig = plt.figure(figsize=(15, 10))

# Вращение вокруг вершины 0 (нижняя левая передняя)
ax1 = fig.add_subplot(1, 2, 1, projection='3d', proj_type='ortho')
vertices_rotated1 = rotate_around_vertex(vertices_cube, 0, [1, 1, 1], np.pi/3)

poly3d_original1 = draw_shape(vertices_cube, faces_cube, 'lightblue', alpha=0.3)
ax1.add_collection3d(poly3d_original1)
poly3d_rotated1 = draw_shape(vertices_rotated1, faces_cube, 'red', alpha=0.7)
ax1.add_collection3d(poly3d_rotated1)

# Отрисовка оси и точки вращения
pivot1 = vertices_cube[:3, 0] / vertices_cube[3, 0]
axis1 = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])
start1 = pivot1 - 1.5 * axis1
end1 = pivot1 + 1.5 * axis1

ax1.quiver(start1[0], start1[1], start1[2], 
           end1[0] - start1[0], end1[1] - start1[1], end1[2] - start1[2],
           color='red', arrow_length_ratio=0.2, linewidth=3)
ax1.scatter([pivot1[0]], [pivot1[1]], [pivot1[2]], color='red', s=100, marker='o')

ax1.set_title('Вращение вокруг вершины 0 (-1,-1,-1)\nна угол 60° вокруг оси [1,1,1]')
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_zlim(-2, 2)
ax1.view_init(azim=-37.5, elev=30)

# Вращение вокруг вершины 6 (верхняя правая задняя)
ax2 = fig.add_subplot(1, 2, 2, projection='3d', proj_type='ortho')
vertices_rotated2 = rotate_around_vertex(vertices_cube, 6, [0, 1, 0], np.pi/2)

poly3d_original2 = draw_shape(vertices_cube, faces_cube, 'lightblue', alpha=0.3)
ax2.add_collection3d(poly3d_original2)
poly3d_rotated2 = draw_shape(vertices_rotated2, faces_cube, 'green', alpha=0.7)
ax2.add_collection3d(poly3d_rotated2)

# Отрисовка оси и точки вращения
pivot2 = vertices_cube[:3, 6] / vertices_cube[3, 6]
axis2 = np.array([0, 1, 0])
start2 = pivot2 - 1.5 * axis2
end2 = pivot2 + 1.5 * axis2

ax2.quiver(start2[0], start2[1], start2[2], 
           end2[0] - start2[0], end2[1] - start2[1], end2[2] - start2[2],
           color='red', arrow_length_ratio=0.2, linewidth=3)
ax2.scatter([pivot2[0]], [pivot2[1]], [pivot2[2]], color='red', s=100, marker='o')

ax2.set_title('Вращение вокруг вершины 6 (1,1,1)\nна угол 90° вокруг оси Y')
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.set_zlim(-2, 2)
ax2.view_init(azim=-37.5, elev=30)

plt.tight_layout()
plt.savefig('images/task5/vertex_rotation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Сравнение вращений вокруг вершин выполнено и сохранено в images/task5/vertex_rotation_comparison.png") 