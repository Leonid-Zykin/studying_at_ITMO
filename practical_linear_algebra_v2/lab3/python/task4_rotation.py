import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from scipy.linalg import expm

# Создаем папку для сохранения изображений
os.makedirs('../images/task4', exist_ok=True)

# Отключаем интерактивный режим matplotlib
plt.ioff()

def create_cube_vertices():
    """Создание вершин кубика в однородных координатах"""
    vertices = np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],    # x координаты
        [-1, -1, 1, 1, -1, -1, 1, 1],    # y координаты
        [-1, -1, -1, -1, 1, 1, 1, 1],    # z координаты
        [1, 1, 1, 1, 1, 1, 1, 1]         # w координаты (однородные)
    ])
    return vertices

def create_cube_faces():
    """Создание граней кубика"""
    faces = np.array([
        [0, 1, 5, 4],  # Передняя грань
        [1, 2, 6, 5],  # Правая грань
        [2, 3, 7, 6],  # Задняя грань
        [3, 0, 4, 7],  # Левая грань
        [0, 1, 2, 3],  # Нижняя грань
        [4, 5, 6, 7]   # Верхняя грань
    ])
    return faces

def create_rotation_matrix(v, theta):
    """Создание матрицы вращения вокруг вектора v на угол theta"""
    # Нормализуем вектор v
    v_norm = v / np.linalg.norm(v)
    vx, vy, vz = v_norm
    
    # Создаем матрицу J
    J = np.array([
        [0, -vz, vy, 0],
        [vz, 0, -vx, 0],
        [-vy, vx, 0, 0],
        [0, 0, 0, 0]
    ])
    
    # Вычисляем матрицу вращения: R = e^(J*theta)
    R = expm(J * theta)
    
    # Устанавливаем последний элемент в 1 для однородных координат
    R[3, 3] = 1
    
    return R

def create_axis_rotation_matrices():
    """Создание матриц вращения вокруг осей координат"""
    # Вращение вокруг оси X
    def Rx(theta):
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]
        ])
    
    # Вращение вокруг оси Y
    def Ry(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ])
    
    # Вращение вокруг оси Z
    def Rz(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    return Rx, Ry, Rz

def apply_transformation(vertices, transformation_matrix):
    """Применение матрицы преобразования к вершинам"""
    return transformation_matrix @ vertices

def draw_cube_with_axis(vertices, faces, rotation_axis, color='blue', 
                       title='Кубик с осью вращения', filename='cube_with_axis'):
    """Отрисовка кубика с осью вращения"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    # Преобразуем однородные координаты в декартовы
    cartesian_vertices = (vertices[:3, :] / vertices[3, :]).T
    
    # Создаем грани для отрисовки
    poly3d = []
    for face in faces:
        poly3d.append([cartesian_vertices[vertex] for vertex in face])
    
    # Отрисовываем кубик
    ax.add_collection3d(Poly3DCollection(poly3d, 
                                        facecolors=color, 
                                        edgecolors='black', 
                                        linewidths=0.5,
                                        alpha=0.7))
    
    # Отрисовываем ось вращения
    axis_length = 3
    axis_points = np.array([
        [-axis_length * rotation_axis[0], axis_length * rotation_axis[0]],
        [-axis_length * rotation_axis[1], axis_length * rotation_axis[1]],
        [-axis_length * rotation_axis[2], axis_length * rotation_axis[2]]
    ])
    
    ax.plot(axis_points[0], axis_points[1], axis_points[2], 
            'r-', linewidth=3, label='Ось вращения')
    
    # Настройка осей
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Устанавливаем одинаковые масштабы по осям
    ax.set_box_aspect([1, 1, 1])
    
    # Динамически устанавливаем пределы осей
    max_range = max(np.max(cartesian_vertices), abs(np.min(cartesian_vertices)))
    ax.set_xlim(-max_range-1, max_range+1)
    ax.set_ylim(-max_range-1, max_range+1)
    ax.set_zlim(-max_range-1, max_range+1)
    
    # Устанавливаем углы обзора
    ax.view_init(azim=-37.5, elev=30)
    
    # Добавляем сетку
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'../images/task4/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_cube(vertices, faces, color='blue', title='Кубик', filename='cube'):
    """Отрисовка кубика"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    # Преобразуем однородные координаты в декартовы
    cartesian_vertices = (vertices[:3, :] / vertices[3, :]).T
    
    # Создаем грани для отрисовки
    poly3d = []
    for face in faces:
        poly3d.append([cartesian_vertices[vertex] for vertex in face])
    
    # Отрисовываем кубик
    ax.add_collection3d(Poly3DCollection(poly3d, 
                                        facecolors=color, 
                                        edgecolors='black', 
                                        linewidths=0.5,
                                        alpha=0.7))
    
    # Настройка осей
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Устанавливаем одинаковые масштабы по осям
    ax.set_box_aspect([1, 1, 1])
    
    # Динамически устанавливаем пределы осей
    max_range = max(np.max(cartesian_vertices), abs(np.min(cartesian_vertices)))
    ax.set_xlim(-max_range-1, max_range+1)
    ax.set_ylim(-max_range-1, max_range+1)
    ax.set_zlim(-max_range-1, max_range+1)
    
    # Устанавливаем углы обзора
    ax.view_init(azim=-37.5, elev=30)
    
    # Добавляем сетку
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../images/task4/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_rotation_effects():
    """Анализ эффектов вращения"""
    print("Анализ эффектов вращения:")
    print("=" * 50)
    
    vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    # Отрисовываем исходный кубик
    draw_cube(vertices, faces, 'blue', 'Исходный кубик', 'original_cube')
    print("Исходный кубик сохранен")
    
    # Различные оси вращения
    rotation_cases = [
        (np.array([1, 0, 0]), np.pi/4, 'red', 'Вращение вокруг оси X на 45°', 'rotate_x'),
        (np.array([0, 1, 0]), np.pi/4, 'green', 'Вращение вокруг оси Y на 45°', 'rotate_y'),
        (np.array([0, 0, 1]), np.pi/4, 'orange', 'Вращение вокруг оси Z на 45°', 'rotate_z'),
        (np.array([1, 1, 0]), np.pi/3, 'purple', 'Вращение вокруг диагонали XY на 60°', 'rotate_xy'),
        (np.array([1, 1, 1]), np.pi/2, 'brown', 'Вращение вокруг диагонали XYZ на 90°', 'rotate_xyz'),
        (np.array([0.5, 1, 0.3]), np.pi/6, 'cyan', 'Вращение вокруг произвольной оси на 30°', 'rotate_arbitrary')
    ]
    
    for axis, angle, color, title, filename in rotation_cases:
        print(f"\nВращение вокруг оси {axis} на угол {angle:.2f} радиан")
        
        # Создаем матрицу вращения
        R = create_rotation_matrix(axis, angle)
        print(f"Матрица вращения R:")
        print(R)
        
        # Применяем преобразование
        transformed_vertices = apply_transformation(vertices, R)
        
        # Отрисовываем результат с осью вращения
        draw_cube_with_axis(transformed_vertices, faces, axis, color, title, filename)
        print(f"Результат сохранен в {filename}.png")
    
    # Сравнительный анализ
    print("\nСравнительный анализ вращения:")
    print("1. Вращение сохраняет форму и размер объекта")
    print("2. Вращение изменяет ориентацию объекта в пространстве")
    print("3. Матрица вращения ортогональна: R^T = R^(-1)")
    print("4. Определитель матрицы вращения равен 1")

def compare_with_axis_rotations():
    """Сравнение с вращениями вокруг осей координат"""
    print("\nСравнение с вращениями вокруг осей координат:")
    print("=" * 50)
    
    vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    # Получаем функции для вращений вокруг осей
    Rx, Ry, Rz = create_axis_rotation_matrices()
    
    # Тестируем вращения вокруг осей координат
    axis_rotations = [
        (Rx, np.pi/4, 'red', 'Вращение вокруг оси X (стандартная матрица)', 'rotate_x_standard'),
        (Ry, np.pi/4, 'green', 'Вращение вокруг оси Y (стандартная матрица)', 'rotate_y_standard'),
        (Rz, np.pi/4, 'orange', 'Вращение вокруг оси Z (стандартная матрица)', 'rotate_z_standard')
    ]
    
    for R_func, angle, color, title, filename in axis_rotations:
        print(f"\nВращение на угол {angle:.2f} радиан")
        
        # Создаем матрицу вращения
        R = R_func(angle)
        print(f"Матрица вращения:")
        print(R)
        
        # Применяем преобразование
        transformed_vertices = apply_transformation(vertices, R)
        
        # Отрисовываем результат
        draw_cube(transformed_vertices, faces, color, title, filename)
        print(f"Результат сохранен в {filename}.png")
    
    # Проверяем эквивалентность методов
    print("\nПроверка эквивалентности методов:")
    
    # Вращение вокруг оси X
    R_x_general = create_rotation_matrix(np.array([1, 0, 0]), np.pi/4)
    R_x_standard = Rx(np.pi/4)
    
    equivalent_x = np.allclose(R_x_general, R_x_standard, atol=1e-10)
    print(f"Вращение вокруг оси X: общий метод == стандартный метод: {equivalent_x}")
    
    # Вращение вокруг оси Y
    R_y_general = create_rotation_matrix(np.array([0, 1, 0]), np.pi/4)
    R_y_standard = Ry(np.pi/4)
    
    equivalent_y = np.allclose(R_y_general, R_y_standard, atol=1e-10)
    print(f"Вращение вокруг оси Y: общий метод == стандартный метод: {equivalent_y}")
    
    # Вращение вокруг оси Z
    R_z_general = create_rotation_matrix(np.array([0, 0, 1]), np.pi/4)
    R_z_standard = Rz(np.pi/4)
    
    equivalent_z = np.allclose(R_z_general, R_z_standard, atol=1e-10)
    print(f"Вращение вокруг оси Z: общий метод == стандартный метод: {equivalent_z}")

def investigate_euler_theorem():
    """Исследование теоремы вращения Эйлера"""
    print("\nИсследование теоремы вращения Эйлера:")
    print("=" * 50)
    
    # Создаем композицию вращений
    Rx, Ry, Rz = create_axis_rotation_matrices()
    
    # Композиция Rx(θ) @ Ry(φ) @ Rz(ψ)
    theta, phi, psi = np.pi/4, np.pi/3, np.pi/6
    R_composite = Rx(theta) @ Ry(phi) @ Rz(psi)
    
    print(f"Композиция Rx({theta:.2f}) @ Ry({phi:.2f}) @ Rz({psi:.2f}):")
    print(R_composite)
    
    # Проверяем свойства
    print(f"\nОпределитель: {np.linalg.det(R_composite):.6f}")
    print(f"Ортогональность: R^T @ R == I: {np.allclose(R_composite.T @ R_composite, np.eye(4))}")
    
    # Извлекаем ось вращения и угол
    # Для матрицы вращения 3x3: trace(R) = 1 + 2*cos(angle)
    R_3x3 = R_composite[:3, :3]
    trace = np.trace(R_3x3)
    angle = np.arccos((trace - 1) / 2)
    
    print(f"Угол вращения: {angle:.4f} радиан ({np.degrees(angle):.2f}°)")
    
    # Находим ось вращения (собственный вектор с собственным числом 1)
    eigenvalues, eigenvectors = np.linalg.eig(R_3x3)
    
    # Ищем собственный вектор с собственным числом, близким к 1
    real_eigenvalues = np.real(eigenvalues)
    axis_idx = np.argmin(np.abs(real_eigenvalues - 1))
    rotation_axis = np.real(eigenvectors[:, axis_idx])
    
    # Нормализуем ось
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    print(f"Ось вращения: {rotation_axis}")
    
    # Проверяем, что можем восстановить матрицу
    R_reconstructed = create_rotation_matrix(rotation_axis, angle)
    reconstruction_error = np.linalg.norm(R_composite - R_reconstructed)
    print(f"Ошибка восстановления: {reconstruction_error:.2e}")
    
    print("\nВыводы:")
    print("1. Любое вращение в 3D можно представить как вращение вокруг одной оси")
    print("2. Композиция Rx(θ)Ry(φ)Rz(ψ) не покрывает все возможные вращения")
    print("3. Для полного покрытия нужны другие параметризации (например, кватернионы)")

def main():
    print("Задание 4: Вращение кубика")
    print("=" * 50)
    
    # Анализируем эффекты вращения
    analyze_rotation_effects()
    
    # Сравниваем с вращениями вокруг осей координат
    compare_with_axis_rotations()
    
    # Исследуем теорему вращения Эйлера
    investigate_euler_theorem()
    
    print("\nЗадание 4 завершено!")
    print("Все изображения сохранены в папке images/task4/")

if __name__ == "__main__":
    main() 