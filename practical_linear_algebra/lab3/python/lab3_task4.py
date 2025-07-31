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

def create_rotation_matrix_x(theta):
    """Создание матрицы вращения вокруг оси X"""
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
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

def create_rotation_matrix_z(theta):
    """Создание матрицы вращения вокруг оси Z"""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
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

def draw_rotation_axis(ax, v, length=2):
    """Отрисовка оси вращения"""
    v_norm = np.array(v) / np.linalg.norm(v)
    start = -length * v_norm
    end = length * v_norm
    
    ax.quiver(start[0], start[1], start[2], 
              end[0] - start[0], end[1] - start[1], end[2] - start[2],
              color='red', arrow_length_ratio=0.2, linewidth=3)

# Создание фигуры для демонстрации вращений
fig = plt.figure(figsize=(20, 15))

# 1. Вращение вокруг произвольной оси
ax1 = fig.add_subplot(2, 3, 1, projection='3d', proj_type='ortho')
v1 = [1, 1, 1]  # Ось вращения
theta1 = np.pi/3  # 60 градусов
R1 = create_rotation_matrix_around_axis(v1, theta1)
vertices_rotated1 = apply_transformation(vertices_cube, R1)

poly3d_original1 = draw_shape(vertices_cube, faces_cube, 'lightblue', alpha=0.3)
ax1.add_collection3d(poly3d_original1)
poly3d_rotated1 = draw_shape(vertices_rotated1, faces_cube, 'red', alpha=0.7)
ax1.add_collection3d(poly3d_rotated1)
draw_rotation_axis(ax1, v1)
ax1.set_title(f'Вращение вокруг оси {v1}\nна угол {theta1*180/np.pi:.0f}°')
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_zlim(-1.5, 1.5)
ax1.view_init(azim=-37.5, elev=30)

# 2. Вращение вокруг оси X
ax2 = fig.add_subplot(2, 3, 2, projection='3d', proj_type='ortho')
theta2 = np.pi/4  # 45 градусов
R2 = create_rotation_matrix_x(theta2)
vertices_rotated2 = apply_transformation(vertices_cube, R2)

poly3d_original2 = draw_shape(vertices_cube, faces_cube, 'lightblue', alpha=0.3)
ax2.add_collection3d(poly3d_original2)
poly3d_rotated2 = draw_shape(vertices_rotated2, faces_cube, 'green', alpha=0.7)
ax2.add_collection3d(poly3d_rotated2)
draw_rotation_axis(ax2, [1, 0, 0])
ax2.set_title(f'Вращение вокруг оси X\nна угол {theta2*180/np.pi:.0f}°')
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_zlim(-1.5, 1.5)
ax2.view_init(azim=-37.5, elev=30)

# 3. Вращение вокруг оси Y
ax3 = fig.add_subplot(2, 3, 3, projection='3d', proj_type='ortho')
theta3 = np.pi/6  # 30 градусов
R3 = create_rotation_matrix_y(theta3)
vertices_rotated3 = apply_transformation(vertices_cube, R3)

poly3d_original3 = draw_shape(vertices_cube, faces_cube, 'lightblue', alpha=0.3)
ax3.add_collection3d(poly3d_original3)
poly3d_rotated3 = draw_shape(vertices_rotated3, faces_cube, 'orange', alpha=0.7)
ax3.add_collection3d(poly3d_rotated3)
draw_rotation_axis(ax3, [0, 1, 0])
ax3.set_title(f'Вращение вокруг оси Y\nна угол {theta3*180/np.pi:.0f}°')
ax3.set_box_aspect([1, 1, 1])
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_zlim(-1.5, 1.5)
ax3.view_init(azim=-37.5, elev=30)

# 4. Вращение вокруг оси Z
ax4 = fig.add_subplot(2, 3, 4, projection='3d', proj_type='ortho')
theta4 = np.pi/2  # 90 градусов
R4 = create_rotation_matrix_z(theta4)
vertices_rotated4 = apply_transformation(vertices_cube, R4)

poly3d_original4 = draw_shape(vertices_cube, faces_cube, 'lightblue', alpha=0.3)
ax4.add_collection3d(poly3d_original4)
poly3d_rotated4 = draw_shape(vertices_rotated4, faces_cube, 'purple', alpha=0.7)
ax4.add_collection3d(poly3d_rotated4)
draw_rotation_axis(ax4, [0, 0, 1])
ax4.set_title(f'Вращение вокруг оси Z\nна угол {theta4*180/np.pi:.0f}°')
ax4.set_box_aspect([1, 1, 1])
ax4.set_xlim(-1.5, 1.5)
ax4.set_ylim(-1.5, 1.5)
ax4.set_zlim(-1.5, 1.5)
ax4.view_init(azim=-37.5, elev=30)

# 5. Композитное вращение Rx(θ)Ry(φ)Rz(ψ)
ax5 = fig.add_subplot(2, 3, 5, projection='3d', proj_type='ortho')
theta_x = np.pi/4
phi_y = np.pi/6
psi_z = np.pi/3
R_composite = create_rotation_matrix_x(theta_x) @ create_rotation_matrix_y(phi_y) @ create_rotation_matrix_z(psi_z)
vertices_rotated5 = apply_transformation(vertices_cube, R_composite)

poly3d_original5 = draw_shape(vertices_cube, faces_cube, 'lightblue', alpha=0.3)
ax5.add_collection3d(poly3d_original5)
poly3d_rotated5 = draw_shape(vertices_rotated5, faces_cube, 'brown', alpha=0.7)
ax5.add_collection3d(poly3d_rotated5)
ax5.set_title(f'Композитное вращение\nRx({theta_x*180/np.pi:.0f}°)Ry({phi_y*180/np.pi:.0f}°)Rz({psi_z*180/np.pi:.0f}°)')
ax5.set_box_aspect([1, 1, 1])
ax5.set_xlim(-1.5, 1.5)
ax5.set_ylim(-1.5, 1.5)
ax5.set_zlim(-1.5, 1.5)
ax5.view_init(azim=-37.5, elev=30)

# 6. Вращение вокруг другой произвольной оси
ax6 = fig.add_subplot(2, 3, 6, projection='3d', proj_type='ortho')
v2 = [1, 0, 1]  # Другая ось вращения
theta6 = np.pi/2  # 90 градусов
R6 = create_rotation_matrix_around_axis(v2, theta6)
vertices_rotated6 = apply_transformation(vertices_cube, R6)

poly3d_original6 = draw_shape(vertices_cube, faces_cube, 'lightblue', alpha=0.3)
ax6.add_collection3d(poly3d_original6)
poly3d_rotated6 = draw_shape(vertices_rotated6, faces_cube, 'darkred', alpha=0.7)
ax6.add_collection3d(poly3d_rotated6)
draw_rotation_axis(ax6, v2)
ax6.set_title(f'Вращение вокруг оси {v2}\nна угол {theta6*180/np.pi:.0f}°')
ax6.set_box_aspect([1, 1, 1])
ax6.set_xlim(-1.5, 1.5)
ax6.set_ylim(-1.5, 1.5)
ax6.set_zlim(-1.5, 1.5)
ax6.view_init(azim=-37.5, elev=30)

plt.tight_layout()
plt.savefig('images/task4/rotation_transformations.png', dpi=300, bbox_inches='tight')
plt.close()

print("Вращения кубика выполнены и сохранены в images/task4/rotation_transformations.png")

# Дополнительная визуализация: сравнение формул вращения
fig = plt.figure(figsize=(15, 10))

# Сравнение нашей формулы с стандартными матрицами вращения
theta = np.pi/3
v = [1, 1, 1]
v_norm = np.array(v) / np.linalg.norm(v)

# Наша формула
R_custom = create_rotation_matrix_around_axis(v, theta)

# Стандартная формула Родрига
def rodrigues_rotation(v, theta):
    v_norm = np.array(v) / np.linalg.norm(v)
    K = np.array([
        [0, -v_norm[2], v_norm[1]],
        [v_norm[2], 0, -v_norm[0]],
        [-v_norm[1], v_norm[0], 0]
    ])
    R_3x3 = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    R_4x4 = np.eye(4)
    R_4x4[:3, :3] = R_3x3
    return R_4x4

R_rodrigues = rodrigues_rotation(v, theta)

# Применение обеих матриц
vertices_custom = apply_transformation(vertices_cube, R_custom)
vertices_rodrigues = apply_transformation(vertices_cube, R_rodrigues)

# Отрисовка сравнения
ax1 = fig.add_subplot(1, 2, 1, projection='3d', proj_type='ortho')
poly3d_custom = draw_shape(vertices_custom, faces_cube, 'red', alpha=0.7)
ax1.add_collection3d(poly3d_custom)
draw_rotation_axis(ax1, v)
ax1.set_title('Наша формула Rv(θ) = e^(Jθ)')
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_zlim(-1.5, 1.5)
ax1.view_init(azim=-37.5, elev=30)

ax2 = fig.add_subplot(1, 2, 2, projection='3d', proj_type='ortho')
poly3d_rodrigues = draw_shape(vertices_rodrigues, faces_cube, 'blue', alpha=0.7)
ax2.add_collection3d(poly3d_rodrigues)
draw_rotation_axis(ax2, v)
ax2.set_title('Формула Родрига')
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_zlim(-1.5, 1.5)
ax2.view_init(azim=-37.5, elev=30)

plt.tight_layout()
plt.savefig('images/task4/rotation_formula_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Сравнение формул вращения выполнено и сохранено в images/task4/rotation_formula_comparison.png") 