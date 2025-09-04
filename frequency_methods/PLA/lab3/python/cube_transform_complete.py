import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# Create directories for images
os.makedirs('../images/task1', exist_ok=True)
os.makedirs('../images/task2', exist_ok=True)
os.makedirs('../images/task3', exist_ok=True)
os.makedirs('../images/task4', exist_ok=True)
os.makedirs('../images/task5', exist_ok=True)
os.makedirs('../images/task6', exist_ok=True)
os.makedirs('../images/task7', exist_ok=True)

# 1. Построение куба

def create_cube():
    # Вершины куба (ребро 2, центр в начале координат)
    return np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1]
    ])

def get_faces():
    return [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4]
    ]

def plot_cube(ax, vertices, faces, color, alpha=0.3, label=None):
    for face in faces:
        ax.add_collection3d(Poly3DCollection([vertices[face]], color=color, alpha=alpha))
    if label:
        ax.scatter([], [], [], color=color, label=label)

def set_axes_equal(ax, vertices):
    """Устанавливает одинаковый масштаб по всем осям и подбирает границы по фигуре."""
    x_limits = [np.min(vertices[:, 0]), np.max(vertices[:, 0])]
    y_limits = [np.min(vertices[:, 1]), np.max(vertices[:, 1])]
    z_limits = [np.min(vertices[:, 2]), np.max(vertices[:, 2])]
    max_range = max(
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0]
    ) / 2.0
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

# 2. Масштабирование

def scale_matrix(sx, sy, sz):
    """Матрица масштабирования"""
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

def apply_transform(vertices, matrix):
    """Применяет 4x4 матрицу к вершинам"""
    # Добавляем однородную координату
    homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    # Применяем преобразование
    transformed = homogeneous @ matrix.T
    # Возвращаем к обычным координатам
    return transformed[:, :3]

# 3. Перенос

def translation_matrix(tx, ty, tz):
    """Матрица переноса"""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

# 4. Повороты вокруг осей

def rotation_x_matrix(theta):
    """Матрица поворота вокруг оси X"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

def rotation_y_matrix(theta):
    """Матрица поворота вокруг оси Y"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])

def rotation_z_matrix(theta):
    """Матрица поворота вокруг оси Z"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# 5. Вращение вокруг произвольной оси (формула Родрига)

def rotation_axis_matrix(v, theta):
    """Матрица поворота вокруг произвольной оси v на угол theta"""
    v = v / np.linalg.norm(v)  # Нормализуем вектор
    vx, vy, vz = v
    
    # Матрица J
    J = np.array([
        [0, -vz, vy, 0],
        [vz, 0, -vx, 0],
        [-vy, vx, 0, 0],
        [0, 0, 0, 0]
    ])
    
    # Формула Родрига: R = I + sin(θ)J + (1-cos(θ))J²
    I = np.eye(4)
    R = I + np.sin(theta) * J + (1 - np.cos(theta)) * (J @ J)
    return R

# 6. Вращение вокруг вершины

def rotate_around_vertex(vertices, vertex_idx, theta, axis_vector):
    """Поворот куба вокруг вершины"""
    v = vertices[vertex_idx]
    
    # Сдвигаем так, чтобы вершина была в начале координат
    T1 = translation_matrix(-v[0], -v[1], -v[2])
    
    # Поворачиваем вокруг оси
    R = rotation_axis_matrix(axis_vector, theta)
    
    # Возвращаем обратно
    T2 = translation_matrix(v[0], v[1], v[2])
    
    # Композиция преобразований
    transform = T2 @ R @ T1
    return apply_transform(vertices, transform)

# 7. Камера

def camera_matrix(tx, ty, tz, rx, ry, rz):
    """Матрица камеры (обратное преобразование)"""
    # Поворот камеры
    R = rotation_z_matrix(rz) @ rotation_y_matrix(ry) @ rotation_x_matrix(rx)
    # Перенос камеры
    T = translation_matrix(-tx, -ty, -tz)
    return T @ R

# 8. Перспективная проекция

def perspective_matrix(fov=60, near=0.1, far=100):
    """Матрица перспективной проекции"""
    f = 1.0 / np.tan(np.radians(fov) / 2)
    return np.array([
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (far - near), -1],
        [0, 0, 2 * far * near / (far - near), 0]
    ])

def apply_perspective(vertices, fov=60, near=0.1, far=100):
    """Применяет перспективную проекцию"""
    P = perspective_matrix(fov, near, far)
    transformed = apply_transform(vertices, P)
    return transformed

# Основная функция для генерации всех изображений

def generate_all_images():
    vertices = create_cube()
    faces = get_faces()
    
    # Задание 1: Построение куба
    print("Генерация изображений для задания 1...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_cube(ax, vertices, faces, 'blue', 0.5, 'Исходный куб')
    ax.set_title('Задание 1: Построение куба в 3D пространстве')
    ax.legend()
    set_axes_equal(ax, vertices)
    plt.savefig('../images/task1/cube_original.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Тетраэдр для демонстрации других фигур
    tetrahedron_vertices = np.array([
        [0, 0, 1.5],
        [1, 0, -0.5],
        [-0.5, 0.866, -0.5],
        [-0.5, -0.866, -0.5]
    ])
    tetrahedron_faces = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_cube(ax, tetrahedron_vertices, tetrahedron_faces, 'red', 0.5, 'Тетраэдр')
    ax.set_title('Пример: Тетраэдр в 3D пространстве')
    ax.legend()
    set_axes_equal(ax, tetrahedron_vertices)
    plt.savefig('../images/task1/tetrahedron.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Задание 2: Масштабирование
    print("Генерация изображений для задания 2...")
    
    # Исходный куб
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    plot_cube(ax1, vertices, faces, 'blue', 0.5)
    ax1.set_title('Исходный куб')
    set_axes_equal(ax1, vertices)
    
    # Масштабированный куб
    S = scale_matrix(2, 1.5, 0.5)
    scaled_vertices = apply_transform(vertices, S)
    ax2 = fig.add_subplot(132, projection='3d')
    plot_cube(ax2, scaled_vertices, faces, 'green', 0.5)
    ax2.set_title('Масштабированный куб\n(2, 1.5, 0.5)')
    set_axes_equal(ax2, scaled_vertices)
    
    # Уменьшенный куб
    S_small = scale_matrix(0.5, 0.5, 0.5)
    small_vertices = apply_transform(vertices, S_small)
    ax3 = fig.add_subplot(133, projection='3d')
    plot_cube(ax3, small_vertices, faces, 'orange', 0.5)
    ax3.set_title('Уменьшенный куб\n(0.5, 0.5, 0.5)')
    set_axes_equal(ax3, small_vertices)
    
    plt.tight_layout()
    plt.savefig('../images/task2/scaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Задание 3: Перенос
    print("Генерация изображений для задания 3...")
    
    fig = plt.figure(figsize=(15, 5))
    
    # Исходный куб
    ax1 = fig.add_subplot(131, projection='3d')
    plot_cube(ax1, vertices, faces, 'blue', 0.5)
    ax1.set_title('Исходный куб')
    set_axes_equal(ax1, vertices)
    
    # Перенесенный куб
    T = translation_matrix(3, 2, 1)
    translated_vertices = apply_transform(vertices, T)
    ax2 = fig.add_subplot(132, projection='3d')
    plot_cube(ax2, translated_vertices, faces, 'green', 0.5)
    ax2.set_title('Перенесенный куб\n(3, 2, 1)')
    set_axes_equal(ax2, translated_vertices)
    
    # Комбинированное преобразование TS
    TS = T @ S
    ts_vertices = apply_transform(vertices, TS)
    ax3 = fig.add_subplot(133, projection='3d')
    plot_cube(ax3, ts_vertices, faces, 'red', 0.5)
    ax3.set_title('Комбинированное TS\n(перенос + масштаб)')
    set_axes_equal(ax3, ts_vertices)
    
    plt.tight_layout()
    plt.savefig('../images/task3/translation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Демонстрация некоммутативности ST vs TS
    ST = S @ T
    st_vertices = apply_transform(vertices, ST)
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    plot_cube(ax1, ts_vertices, faces, 'red', 0.5)
    ax1.set_title('TS (перенос, затем масштаб)')
    set_axes_equal(ax1, ts_vertices)
    
    ax2 = fig.add_subplot(132, projection='3d')
    plot_cube(ax2, st_vertices, faces, 'purple', 0.5)
    ax2.set_title('ST (масштаб, затем перенос)')
    set_axes_equal(ax2, st_vertices)
    
    # Оба куба на одном графике
    ax3 = fig.add_subplot(133, projection='3d')
    plot_cube(ax3, ts_vertices, faces, 'red', 0.3, 'TS')
    plot_cube(ax3, st_vertices, faces, 'purple', 0.3, 'ST')
    ax3.set_title('Сравнение TS и ST')
    ax3.legend()
    all_vertices = np.vstack([ts_vertices, st_vertices])
    set_axes_equal(ax3, all_vertices)
    
    plt.tight_layout()
    plt.savefig('../images/task3/ts_vs_st.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Задание 4: Поворот вокруг произвольной оси
    print("Генерация изображений для задания 4...")
    
    # Вектор оси вращения
    v = np.array([1, 1, 1])
    theta = np.pi/3
    
    # Поворот вокруг произвольной оси
    R_axis = rotation_axis_matrix(v, theta)
    rotated_vertices = apply_transform(vertices, R_axis)
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    plot_cube(ax1, vertices, faces, 'blue', 0.5)
    ax1.set_title('Исходный куб')
    set_axes_equal(ax1, vertices)
    
    ax2 = fig.add_subplot(132, projection='3d')
    plot_cube(ax2, rotated_vertices, faces, 'green', 0.5)
    # Добавляем вектор оси
    ax2.quiver(0, 0, 0, v[0], v[1], v[2], color='red', linewidth=3, label='Ось вращения')
    ax2.set_title(f'Поворот вокруг оси {v}\nна угол {theta:.2f} рад')
    ax2.legend()
    set_axes_equal(ax2, rotated_vertices)
    
    # Повороты вокруг координатных осей
    R_x = rotation_x_matrix(np.pi/4)
    R_y = rotation_y_matrix(np.pi/4)
    R_z = rotation_z_matrix(np.pi/4)
    
    # Композиция поворотов
    R_xyz = R_x @ R_y @ R_z
    xyz_vertices = apply_transform(vertices, R_xyz)
    
    ax3 = fig.add_subplot(133, projection='3d')
    plot_cube(ax3, xyz_vertices, faces, 'purple', 0.5)
    ax3.set_title('Rx(π/4)Ry(π/4)Rz(π/4)')
    set_axes_equal(ax3, xyz_vertices)
    
    plt.tight_layout()
    plt.savefig('../images/task4/rotation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Задание 5: Поворот вокруг вершины
    print("Генерация изображений для задания 5...")
    
    # Поворот вокруг вершины 0
    vertex_idx = 0
    axis_vector = np.array([0, 0, 1])  # Вокруг оси Z
    theta = np.pi/3
    
    rotated_around_vertex = rotate_around_vertex(vertices, vertex_idx, theta, axis_vector)
    
    fig = plt.figure(figsize=(12, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    plot_cube(ax1, vertices, faces, 'blue', 0.3, 'Исходный куб')
    plot_cube(ax1, rotated_around_vertex, faces, 'red', 0.3, 'Повернутый куб')
    ax1.set_title('Поворот куба вокруг вершины 0\nна угол π/3 вокруг оси Z')
    ax1.legend()
    all_vertices = np.vstack([vertices, rotated_around_vertex])
    set_axes_equal(ax1, all_vertices)
    
    # Другой угол поворота
    rotated_around_vertex2 = rotate_around_vertex(vertices, vertex_idx, np.pi/2, axis_vector)
    
    ax2 = fig.add_subplot(122, projection='3d')
    plot_cube(ax2, vertices, faces, 'blue', 0.3, 'Исходный куб')
    plot_cube(ax2, rotated_around_vertex2, faces, 'green', 0.3, 'Повернутый куб')
    ax2.set_title('Поворот куба вокруг вершины 0\nна угол π/2 вокруг оси Z')
    ax2.legend()
    all_vertices2 = np.vstack([vertices, rotated_around_vertex2])
    set_axes_equal(ax2, all_vertices2)
    
    plt.tight_layout()
    plt.savefig('../images/task5/rotation_around_vertex.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Задание 6: Виртуальная камера
    print("Генерация изображений для задания 6...")
    
    # Создаем сцену с несколькими кубами
    scene_vertices = []
    scene_faces = []
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    positions = [
        (0, 0, 0),
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
        (3, 3, 3)
    ]
    
    for i, (dx, dy, dz) in enumerate(positions):
        T_cube = translation_matrix(dx, dy, dz)
        cube_vertices = apply_transform(vertices, T_cube)
        scene_vertices.append(cube_vertices)
        scene_faces.extend(faces)
    
    scene_vertices = np.vstack(scene_vertices)
    
    # Исходная сцена
    fig = plt.figure(figsize=(15, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(len(positions)):
        start_idx = i * len(faces)
        end_idx = (i + 1) * len(faces)
        cube_faces = [[f[j] - i * 8 for j in range(len(f))] for f in faces]
        plot_cube(ax1, scene_vertices[i*8:(i+1)*8], cube_faces, colors[i], 0.5)
    ax1.set_title('Исходная сцена с несколькими кубами')
    set_axes_equal(ax1, scene_vertices)
    
    # Применяем камеру
    camera_transform_matrix = camera_matrix(2, 2, 2, np.pi/6, np.pi/6, 0)
    camera_vertices = apply_transform(scene_vertices, camera_transform_matrix)
    
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(len(positions)):
        start_idx = i * len(faces)
        end_idx = (i + 1) * len(faces)
        cube_faces = [[f[j] - i * 8 for j in range(len(f))] for f in faces]
        plot_cube(ax2, camera_vertices[i*8:(i+1)*8], cube_faces, colors[i], 0.5)
    ax2.set_title('Сцена после применения камеры\n(позиция: 2,2,2, поворот: π/6,π/6,0)')
    set_axes_equal(ax2, camera_vertices)
    
    plt.tight_layout()
    plt.savefig('../images/task6/camera_scene.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Задание 7: Перспективная проекция
    print("Генерация изображений для задания 7...")
    
    # Применяем перспективную проекцию к сцене
    perspective_vertices = apply_perspective(scene_vertices, fov=60, near=0.1, far=100)
    
    fig = plt.figure(figsize=(15, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(len(positions)):
        cube_faces = [[f[j] - i * 8 for j in range(len(f))] for f in faces]
        plot_cube(ax1, scene_vertices[i*8:(i+1)*8], cube_faces, colors[i], 0.5)
    ax1.set_title('Сцена без перспективы')
    set_axes_equal(ax1, scene_vertices)
    
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(len(positions)):
        cube_faces = [[f[j] - i * 8 for j in range(len(f))] for f in faces]
        plot_cube(ax2, perspective_vertices[i*8:(i+1)*8], cube_faces, colors[i], 0.5)
    ax2.set_title('Сцена с перспективной проекцией\n(fov=60°, near=0.1, far=100)')
    set_axes_equal(ax2, perspective_vertices)
    
    plt.tight_layout()
    plt.savefig('../images/task7/perspective_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Дополнительное изображение: вид сбоку для демонстрации перспективы
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(positions)):
        cube_faces = [[f[j] - i * 8 for j in range(len(f))] for f in faces]
        plot_cube(ax, perspective_vertices[i*8:(i+1)*8], cube_faces, colors[i], 0.5)
    ax.set_title('Перспективная проекция - вид сбоку')
    ax.view_init(azim=0, elev=-90)  # Вид снизу
    set_axes_equal(ax, perspective_vertices)
    plt.savefig('../images/task7/perspective_side_view.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Все изображения успешно сгенерированы!")

if __name__ == '__main__':
    generate_all_images()
