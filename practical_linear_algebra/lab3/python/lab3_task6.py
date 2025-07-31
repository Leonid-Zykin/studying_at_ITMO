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

def create_translation_matrix(tx, ty, tz):
    """Создание матрицы перемещения"""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def create_rotation_matrix_y(theta):
    """Создание матрицы вращения вокруг оси Y"""
    return np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def create_rotation_matrix_x(theta):
    """Создание матрицы вращения вокруг оси X"""
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def create_camera_matrix(camera_position, camera_target, up_vector=[0, 1, 0]):
    """
    Создание матрицы камеры (матрица вида)
    camera_position - позиция камеры
    camera_target - точка, на которую смотрит камера
    up_vector - вектор "вверх" для камеры
    """
    # Вычисление базисных векторов камеры
    forward = np.array(camera_target) - np.array(camera_position)
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, np.array(up_vector))
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    
    # Создание матрицы поворота камеры
    R = np.eye(4)
    R[:3, 0] = right
    R[:3, 1] = up
    R[:3, 2] = -forward
    
    # Создание матрицы перемещения камеры
    T = create_translation_matrix(-camera_position[0], -camera_position[1], -camera_position[2])
    
    # Матрица камеры = R * T
    return R @ T

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

# Создание сцены с несколькими кубиками
scene_objects = [
    # (позиция, цвет, масштаб, поворот)
    ([0, 0, 0], 'blue', 1.0, 0),           # Центральный кубик
    ([2, 0, 0], 'red', 0.8, np.pi/6),      # Кубик справа
    ([-2, 0, 0], 'green', 1.2, -np.pi/4),  # Кубик слева
    ([0, 2, 0], 'orange', 0.9, np.pi/3),   # Кубик сзади
    ([0, -2, 0], 'purple', 1.1, -np.pi/6), # Кубик спереди
    ([1.5, 1.5, 0], 'brown', 0.7, np.pi/2),    # Кубик по диагонали
    ([-1.5, -1.5, 0], 'pink', 0.6, -np.pi/3)   # Кубик по диагонали
]

def create_scaling_matrix(sx, sy, sz):
    """Создание матрицы масштабирования"""
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

# Создание фигуры для демонстрации камеры
fig = plt.figure(figsize=(20, 15))

# 1. Исходная сцена без камеры
ax1 = fig.add_subplot(2, 3, 1, projection='3d', proj_type='ortho')

for pos, color, scale, rotation in scene_objects:
    # Создание матриц преобразования
    T = create_translation_matrix(pos[0], pos[1], pos[2])
    S = create_scaling_matrix(scale, scale, scale)
    R = create_rotation_matrix_y(rotation)
    
    # Применение преобразований
    transformed_vertices = apply_transformation(apply_transformation(apply_transformation(vertices_cube, S), R), T)
    
    # Отрисовка кубика
    poly3d = draw_shape(transformed_vertices, faces_cube, color, alpha=0.7)
    ax1.add_collection3d(poly3d)

ax1.set_title('Исходная сцена\n(вид сверху)')
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_zlim(-2, 2)
ax1.view_init(azim=0, elev=-90)  # Вид снизу

# 2. Сцена с камерой в позиции (5, 5, 3)
ax2 = fig.add_subplot(2, 3, 2, projection='3d', proj_type='ortho')

camera_pos1 = [5, 5, 3]
camera_target1 = [0, 0, 0]
camera_matrix1 = create_camera_matrix(camera_pos1, camera_target1)

for pos, color, scale, rotation in scene_objects:
    T = create_translation_matrix(pos[0], pos[1], pos[2])
    S = create_scaling_matrix(scale, scale, scale)
    R = create_rotation_matrix_y(rotation)
    
    # Применение преобразований и камеры
    transformed_vertices = apply_transformation(apply_transformation(apply_transformation(vertices_cube, S), R), T)
    camera_transformed_vertices = apply_transformation(transformed_vertices, camera_matrix1)
    
    poly3d = draw_shape(camera_transformed_vertices, faces_cube, color, alpha=0.7)
    ax2.add_collection3d(poly3d)

ax2.set_title(f'Камера в позиции {camera_pos1}\nсмотрит на центр')
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_zlim(-3, 3)
ax2.view_init(azim=-37.5, elev=30)

# 3. Сцена с камерой в позиции (-4, -4, 2)
ax3 = fig.add_subplot(2, 3, 3, projection='3d', proj_type='ortho')

camera_pos2 = [-4, -4, 2]
camera_target2 = [0, 0, 0]
camera_matrix2 = create_camera_matrix(camera_pos2, camera_target2)

for pos, color, scale, rotation in scene_objects:
    T = create_translation_matrix(pos[0], pos[1], pos[2])
    S = create_scaling_matrix(scale, scale, scale)
    R = create_rotation_matrix_y(rotation)
    
    transformed_vertices = apply_transformation(apply_transformation(apply_transformation(vertices_cube, S), R), T)
    camera_transformed_vertices = apply_transformation(transformed_vertices, camera_matrix2)
    
    poly3d = draw_shape(camera_transformed_vertices, faces_cube, color, alpha=0.7)
    ax3.add_collection3d(poly3d)

ax3.set_title(f'Камера в позиции {camera_pos2}\nсмотрит на центр')
ax3.set_box_aspect([1, 1, 1])
ax3.set_xlim(-3, 3)
ax3.set_ylim(-3, 3)
ax3.set_zlim(-3, 3)
ax3.view_init(azim=-37.5, elev=30)

# 4. Сцена с камерой высоко над сценой
ax4 = fig.add_subplot(2, 3, 4, projection='3d', proj_type='ortho')

camera_pos3 = [0, 0, 8]
camera_target3 = [0, 0, 0]
camera_matrix3 = create_camera_matrix(camera_pos3, camera_target3)

for pos, color, scale, rotation in scene_objects:
    T = create_translation_matrix(pos[0], pos[1], pos[2])
    S = create_scaling_matrix(scale, scale, scale)
    R = create_rotation_matrix_y(rotation)
    
    transformed_vertices = apply_transformation(apply_transformation(apply_transformation(vertices_cube, S), R), T)
    camera_transformed_vertices = apply_transformation(transformed_vertices, camera_matrix3)
    
    poly3d = draw_shape(camera_transformed_vertices, faces_cube, color, alpha=0.7)
    ax4.add_collection3d(poly3d)

ax4.set_title(f'Камера в позиции {camera_pos3}\n(вид сверху)')
ax4.set_box_aspect([1, 1, 1])
ax4.set_xlim(-3, 3)
ax4.set_ylim(-3, 3)
ax4.set_zlim(-3, 3)
ax4.view_init(azim=-37.5, elev=30)

# 5. Сцена с камерой под углом
ax5 = fig.add_subplot(2, 3, 5, projection='3d', proj_type='ortho')

camera_pos4 = [6, -6, 4]
camera_target4 = [0, 0, 0]
camera_matrix4 = create_camera_matrix(camera_pos4, camera_target4)

for pos, color, scale, rotation in scene_objects:
    T = create_translation_matrix(pos[0], pos[1], pos[2])
    S = create_scaling_matrix(scale, scale, scale)
    R = create_rotation_matrix_y(rotation)
    
    transformed_vertices = apply_transformation(apply_transformation(apply_transformation(vertices_cube, S), R), T)
    camera_transformed_vertices = apply_transformation(transformed_vertices, camera_matrix4)
    
    poly3d = draw_shape(camera_transformed_vertices, faces_cube, color, alpha=0.7)
    ax5.add_collection3d(poly3d)

ax5.set_title(f'Камера в позиции {camera_pos4}\n(вид под углом)')
ax5.set_box_aspect([1, 1, 1])
ax5.set_xlim(-3, 3)
ax5.set_ylim(-3, 3)
ax5.set_zlim(-3, 3)
ax5.view_init(azim=-37.5, elev=30)

# 6. Сравнение с обычным видом
ax6 = fig.add_subplot(2, 3, 6, projection='3d', proj_type='ortho')

for pos, color, scale, rotation in scene_objects:
    T = create_translation_matrix(pos[0], pos[1], pos[2])
    S = create_scaling_matrix(scale, scale, scale)
    R = create_rotation_matrix_y(rotation)
    
    transformed_vertices = apply_transformation(apply_transformation(apply_transformation(vertices_cube, S), R), T)
    
    poly3d = draw_shape(transformed_vertices, faces_cube, color, alpha=0.7)
    ax6.add_collection3d(poly3d)

ax6.set_title('Обычный вид\n(без камеры)')
ax6.set_box_aspect([1, 1, 1])
ax6.set_xlim(-3, 3)
ax6.set_ylim(-3, 3)
ax6.set_zlim(-2, 2)
ax6.view_init(azim=-37.5, elev=30)

plt.tight_layout()
plt.savefig('images/task6/camera_implementations.png', dpi=300, bbox_inches='tight')
plt.close()

print("Реализация камеры выполнена и сохранена в images/task6/camera_implementations.png")

# Дополнительная визуализация: эффект камеры
fig = plt.figure(figsize=(15, 10))

# Сцена без камеры
ax1 = fig.add_subplot(1, 2, 1, projection='3d', proj_type='ortho')

for pos, color, scale, rotation in scene_objects:
    T = create_translation_matrix(pos[0], pos[1], pos[2])
    S = create_scaling_matrix(scale, scale, scale)
    R = create_rotation_matrix_y(rotation)
    
    transformed_vertices = apply_transformation(apply_transformation(apply_transformation(vertices_cube, S), R), T)
    
    poly3d = draw_shape(transformed_vertices, faces_cube, color, alpha=0.7)
    ax1.add_collection3d(poly3d)

ax1.set_title('Сцена без камеры\n(стандартный вид)')
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_zlim(-2, 2)
ax1.view_init(azim=-37.5, elev=30)

# Сцена с камерой
ax2 = fig.add_subplot(1, 2, 2, projection='3d', proj_type='ortho')

camera_pos = [8, -8, 6]
camera_target = [0, 0, 0]
camera_matrix = create_camera_matrix(camera_pos, camera_target)

for pos, color, scale, rotation in scene_objects:
    T = create_translation_matrix(pos[0], pos[1], pos[2])
    S = create_scaling_matrix(scale, scale, scale)
    R = create_rotation_matrix_y(rotation)
    
    transformed_vertices = apply_transformation(apply_transformation(apply_transformation(vertices_cube, S), R), T)
    camera_transformed_vertices = apply_transformation(transformed_vertices, camera_matrix)
    
    poly3d = draw_shape(camera_transformed_vertices, faces_cube, color, alpha=0.7)
    ax2.add_collection3d(poly3d)

ax2.set_title(f'Сцена с камерой\n(позиция {camera_pos})')
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_zlim(-3, 3)
ax2.view_init(azim=-37.5, elev=30)

plt.tight_layout()
plt.savefig('images/task6/camera_effect_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Сравнение эффектов камеры выполнено и сохранено в images/task6/camera_effect_comparison.png") 