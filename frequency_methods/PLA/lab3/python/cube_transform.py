import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

def scale(vertices, sx, sy, sz):
    S = np.diag([sx, sy, sz])
    return vertices @ S.T

# 3. Перенос

def translate(vertices, tx, ty, tz):
    return vertices + np.array([tx, ty, tz])

# 4. Повороты вокруг осей

def rotate_x(vertices, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    return vertices @ R.T

def rotate_y(vertices, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    return vertices @ R.T

def rotate_z(vertices, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    return vertices @ R.T

# 5. Вращение вокруг вершины

def rotate_around_vertex(vertices, faces, vertex_idx, theta, axis='z'):
    v = vertices[vertex_idx]
    moved = vertices - v
    if axis == 'x':
        rotated = rotate_x(moved, theta)
    elif axis == 'y':
        rotated = rotate_y(moved, theta)
    else:
        rotated = rotate_z(moved, theta)
    return rotated + v

# 6. Камера (имитация): просто сдвиг и поворот всей сцены

def camera_transform(vertices, tx=0, ty=0, tz=0, rx=0, ry=0, rz=0):
    v = rotate_x(vertices, rx)
    v = rotate_y(v, ry)
    v = rotate_z(v, rz)
    v = translate(v, tx, ty, tz)
    return v

# 7. Перспектива (простая)

def perspective_projection(vertices, d=5):
    # d — расстояние до камеры
    projected = []
    for x, y, z in vertices:
        if z + d != 0:
            projected.append([x * d / (z + d), y * d / (z + d), 0])
        else:
            projected.append([x, y, 0])
    return np.array(projected)

# 8. Сохранение графиков

def save_figure(fig, filename):
    fig.savefig(filename, bbox_inches='tight')

# Пример использования всех этапов
if __name__ == '__main__':
    vertices = create_cube()
    faces = get_faces()

    # 1. Построение куба
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_cube(ax, vertices, faces, 'blue', 0.5)
    ax.set_title('Исходный куб')
    set_axes_equal(ax, vertices)
    plt.show()
    save_figure(fig, 'cube_original.png')

    # 2. Масштабирование
    scaled = scale(vertices, 2, 1.5, 0.5)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_cube(ax, scaled, faces, 'green', 0.5)
    ax.set_title('Масштабированный куб')
    set_axes_equal(ax, scaled)
    plt.show()
    save_figure(fig, 'cube_scaled.png')

    # 3. Перенос
    moved = translate(vertices, 3, 2, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_cube(ax, moved, faces, 'orange', 0.5)
    ax.set_title('Перенесённый куб')
    set_axes_equal(ax, moved)
    plt.show()
    save_figure(fig, 'cube_moved.png')

    # 4. Поворот
    rotated = rotate_z(vertices, np.pi/4)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_cube(ax, rotated, faces, 'red', 0.5)
    ax.set_title('Поворот куба вокруг Z (45°)')
    set_axes_equal(ax, rotated)
    plt.show()
    save_figure(fig, 'cube_rotated_z.png')

    # 5. Вращение вокруг вершины
    around_vertex = rotate_around_vertex(vertices, faces, 0, np.pi/4, axis='z')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_cube(ax, around_vertex, faces, 'purple', 0.5)
    ax.set_title('Вращение вокруг вершины 0 (Z, 45°)')
    set_axes_equal(ax, around_vertex)
    plt.show()
    save_figure(fig, 'cube_around_vertex.png')

    # 6. Камера (сдвиг и поворот всей сцены)
    cam = camera_transform(vertices, tx=2, ty=2, tz=2, rx=np.pi/8, ry=np.pi/8, rz=0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_cube(ax, cam, faces, 'brown', 0.5)
    ax.set_title('Куб с виртуальной камерой')
    set_axes_equal(ax, cam)
    plt.show()
    save_figure(fig, 'cube_camera.png')

    # 7. Перспектива
    persp = perspective_projection(vertices, d=5)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_cube(ax, persp, faces, 'cyan', 0.5)
    ax.set_title('Перспективная проекция куба')
    set_axes_equal(ax, persp)
    plt.show()
    save_figure(fig, 'cube_perspective.png')

    # 7. Перспектива — несколько повернутых и смещённых кубиков
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['cyan', 'magenta', 'yellow', 'blue', 'orange', 'lime', 'pink', 'deepskyblue']
    all_proj_vertices = []
    for i, (dx, dy, dz, angle) in enumerate([
        (0, 0, 0, 0),
        (2, 2, 2, np.pi/6),
        (-2, 2, 2, np.pi/4),
        (2, -2, 2, np.pi/3),
        (-2, -2, 2, np.pi/2),
        (2, 2, -2, np.pi/5),
        (-2, 2, -2, np.pi/7),
        (2, -2, -2, np.pi/8)
    ]):
        # Повернуть и сместить куб
        v = rotate_z(vertices, angle)
        v = translate(v, dx, dy, dz)
        # Применить перспективу
        v_proj = perspective_projection(v, d=8)
        plot_cube(ax, v_proj, faces, colors[i % len(colors)], 0.4)
        all_proj_vertices.append(v_proj)
    all_proj_vertices = np.vstack(all_proj_vertices)
    ax.set_title('Несколько кубиков с перспективой')
    set_axes_equal(ax, all_proj_vertices)
    plt.show()
    save_figure(fig, 'cubes_perspective_multi.png')

    # 7b. Несколько кубиков в обычном 3D (без перспективы)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    all_vertices = []
    for i, (dx, dy, dz, angle) in enumerate([
        (0, 0, 0, 0),
        (2, 2, 2, np.pi/6),
        (-2, 2, 2, np.pi/4),
        (2, -2, 2, np.pi/3),
        (-2, -2, 2, np.pi/2),
        (2, 2, -2, np.pi/5),
        (-2, 2, -2, np.pi/7),
        (2, -2, -2, np.pi/8)
    ]):
        v = rotate_z(vertices, angle)
        v = translate(v, dx, dy, dz)
        plot_cube(ax, v, faces, colors[i % len(colors)], 0.4)
        all_vertices.append(v)
    all_vertices = np.vstack(all_vertices)
    ax.set_title('Несколько кубиков в 3D (без перспективы)')
    set_axes_equal(ax, all_vertices)
    plt.show()
    save_figure(fig, 'cubes_3d_multi.png') 