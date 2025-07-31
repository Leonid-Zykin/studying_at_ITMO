import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from scipy.linalg import expm

# Создаем папку для сохранения изображений
os.makedirs('../images/task6', exist_ok=True)

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

def create_translation_matrix(tx, ty, tz):
    """Создание матрицы перемещения"""
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return T

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

def apply_transformation(vertices, transformation_matrix):
    """Применение матрицы преобразования к вершинам"""
    return transformation_matrix @ vertices

def create_scene():
    """Создание сцены с несколькими кубиками"""
    print("Создание сцены с несколькими кубиками:")
    print("=" * 50)
    
    # Создаем базовый кубик
    base_vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    # Создаем несколько кубиков в разных позициях
    scene_objects = []
    
    # Кубик 1: в начале координат
    cube1 = {
        'vertices': base_vertices.copy(),
        'color': 'blue',
        'name': 'Кубик 1 (начало координат)'
    }
    scene_objects.append(cube1)
    
    # Кубик 2: смещен по X
    T2 = create_translation_matrix(3, 0, 0)
    cube2 = {
        'vertices': apply_transformation(base_vertices, T2),
        'color': 'red',
        'name': 'Кубик 2 (смещен по X)'
    }
    scene_objects.append(cube2)
    
    # Кубик 3: смещен по Y и повернут
    T3 = create_translation_matrix(0, 3, 0)
    R3 = create_rotation_matrix(np.array([0, 0, 1]), np.pi/4)
    cube3 = {
        'vertices': apply_transformation(apply_transformation(base_vertices, R3), T3),
        'color': 'green',
        'name': 'Кубик 3 (смещен по Y, повернут)'
    }
    scene_objects.append(cube3)
    
    # Кубик 4: смещен по Z и масштабирован
    T4 = create_translation_matrix(0, 0, 3)
    S4 = np.array([
        [1.5, 0, 0, 0],
        [0, 1.5, 0, 0],
        [0, 0, 1.5, 0],
        [0, 0, 0, 1]
    ])
    cube4 = {
        'vertices': apply_transformation(apply_transformation(base_vertices, S4), T4),
        'color': 'orange',
        'name': 'Кубик 4 (смещен по Z, увеличен)'
    }
    scene_objects.append(cube4)
    
    # Кубик 5: в диагональной позиции
    T5 = create_translation_matrix(2, 2, 2)
    R5 = create_rotation_matrix(np.array([1, 1, 0]), np.pi/3)
    cube5 = {
        'vertices': apply_transformation(apply_transformation(base_vertices, R5), T5),
        'color': 'purple',
        'name': 'Кубик 5 (диагональная позиция)'
    }
    scene_objects.append(cube5)
    
    return scene_objects, faces

def draw_scene(scene_objects, faces, title='Сцена', filename='scene'):
    """Отрисовка сцены с несколькими кубиками"""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    # Отрисовываем каждый кубик
    for i, obj in enumerate(scene_objects):
        vertices = obj['vertices']
        color = obj['color']
        
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
    
    # Устанавливаем пределы осей
    ax.set_xlim(-2, 5)
    ax.set_ylim(-2, 5)
    ax.set_zlim(-2, 5)
    
    # Устанавливаем углы обзора
    ax.view_init(azim=-37.5, elev=30)
    
    # Добавляем сетку
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../images/task6/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_camera_matrix(camera_position, camera_target, up_vector):
    """Создание матрицы камеры"""
    # Вычисляем направление камеры
    forward = camera_target - camera_position
    forward = forward / np.linalg.norm(forward)
    
    # Вычисляем правый вектор
    right = np.cross(forward, up_vector)
    right = right / np.linalg.norm(right)
    
    # Пересчитываем up вектор для ортогональности
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Создаем матрицу поворота камеры
    R_camera = np.array([
        [right[0], right[1], right[2], 0],
        [up[0], up[1], up[2], 0],
        [-forward[0], -forward[1], -forward[2], 0],
        [0, 0, 0, 1]
    ])
    
    # Создаем матрицу перемещения камеры
    T_camera = create_translation_matrix(-camera_position[0], -camera_position[1], -camera_position[2])
    
    # Композиция: R_camera @ T_camera
    C = R_camera @ T_camera
    
    return C, R_camera, T_camera

def apply_camera_transformation(scene_objects, camera_matrix):
    """Применение преобразования камеры ко всем объектам сцены"""
    transformed_objects = []
    
    for obj in scene_objects:
        transformed_vertices = apply_transformation(obj['vertices'], camera_matrix)
        transformed_obj = {
            'vertices': transformed_vertices,
            'color': obj['color'],
            'name': obj['name']
        }
        transformed_objects.append(transformed_obj)
    
    return transformed_objects

def analyze_camera_effects():
    """Анализ эффектов камеры"""
    print("Анализ эффектов камеры:")
    print("=" * 50)
    
    # Создаем сцену
    scene_objects, faces = create_scene()
    
    # Отрисовываем исходную сцену
    draw_scene(scene_objects, faces, 'Исходная сцена', 'original_scene')
    print("Исходная сцена сохранена")
    
    # Различные позиции камеры
    camera_positions = [
        (np.array([5, 5, 5]), np.array([0, 0, 0]), np.array([0, 0, 1]), 
         'Камера сбоку и сверху', 'camera_side_top'),
        (np.array([0, 0, 8]), np.array([0, 0, 0]), np.array([0, 1, 0]), 
         'Камера спереди', 'camera_front'),
        (np.array([8, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 1]), 
         'Камера справа', 'camera_right'),
        (np.array([5, 5, 2]), np.array([2, 2, 2]), np.array([0, 0, 1]), 
         'Камера смотрит на центр', 'camera_look_at_center'),
        (np.array([3, 3, 3]), np.array([0, 0, 0]), np.array([0, 0, 1]), 
         'Камера под углом', 'camera_angle')
    ]
    
    for camera_pos, camera_target, up_vector, title, filename in camera_positions:
        print(f"\n{title}")
        print(f"Позиция камеры: {camera_pos}")
        print(f"Цель камеры: {camera_target}")
        
        # Создаем матрицу камеры
        C, R_camera, T_camera = create_camera_matrix(camera_pos, camera_target, up_vector)
        
        print(f"Матрица поворота камеры R_camera:")
        print(R_camera)
        print(f"Матрица перемещения камеры T_camera:")
        print(T_camera)
        print(f"Полная матрица камеры C:")
        print(C)
        
        # Применяем преобразование камеры
        transformed_objects = apply_camera_transformation(scene_objects, C)
        
        # Отрисовываем результат
        draw_scene(transformed_objects, faces, f'Сцена: {title}', filename)
        print(f"Результат сохранен в {filename}.png")
        
        # Анализируем эффект
        print(f"Эффект: камера перемещена в начало координат и повернута")
        print(f"Объекты сцены преобразованы относительно камеры")

def demonstrate_camera_inverse():
    """Демонстрация обратного преобразования камеры"""
    print("\nДемонстрация обратного преобразования камеры:")
    print("=" * 50)
    
    # Создаем сцену
    scene_objects, faces = create_scene()
    
    # Выбираем позицию камеры
    camera_pos = np.array([5, 5, 5])
    camera_target = np.array([0, 0, 0])
    up_vector = np.array([0, 0, 1])
    
    print(f"Позиция камеры: {camera_pos}")
    print(f"Цель камеры: {camera_target}")
    
    # Создаем матрицу камеры
    C, R_camera, T_camera = create_camera_matrix(camera_pos, camera_target, up_vector)
    
    print(f"Матрица камеры C:")
    print(C)
    
    # Вычисляем обратную матрицу
    C_inv = np.linalg.inv(C)
    print(f"Обратная матрица C^(-1):")
    print(C_inv)
    
    # Применяем преобразование камеры
    transformed_objects = apply_camera_transformation(scene_objects, C)
    
    # Применяем обратное преобразование
    restored_objects = apply_camera_transformation(transformed_objects, C_inv)
    
    # Отрисовываем результаты
    draw_scene(transformed_objects, faces, 'Сцена после преобразования камеры', 'camera_transformed')
    draw_scene(restored_objects, faces, 'Сцена после обратного преобразования', 'camera_restored')
    
    # Проверяем восстановление
    restoration_error = 0
    for i, (original, restored) in enumerate(zip(scene_objects, restored_objects)):
        error = np.linalg.norm(original['vertices'] - restored['vertices'])
        restoration_error += error
        print(f"Ошибка восстановления кубика {i+1}: {error:.2e}")
    
    print(f"Общая ошибка восстановления: {restoration_error:.2e}")
    print("Вывод: обратное преобразование камеры восстанавливает исходную сцену")

def investigate_camera_properties():
    """Исследование свойств матрицы камеры"""
    print("\nИсследование свойств матрицы камеры:")
    print("=" * 50)
    
    # Создаем матрицу камеры
    camera_pos = np.array([3, 3, 3])
    camera_target = np.array([0, 0, 0])
    up_vector = np.array([0, 0, 1])
    
    C, R_camera, T_camera = create_camera_matrix(camera_pos, camera_target, up_vector)
    
    print("Свойства матрицы камеры:")
    print(f"Определитель C: {np.linalg.det(C):.6f}")
    print(f"Определитель R_camera: {np.linalg.det(R_camera):.6f}")
    print(f"Определитель T_camera: {np.linalg.det(T_camera):.6f}")
    
    # Проверяем ортогональность матрицы поворота
    orthogonality_check = np.allclose(R_camera @ R_camera.T, np.eye(4))
    print(f"R_camera ортогональна: {orthogonality_check}")
    
    # Проверяем композицию
    composition_check = np.allclose(C, R_camera @ T_camera)
    print(f"C == R_camera @ T_camera: {composition_check}")
    
    # Анализируем эффект на координаты
    print(f"\nЭффект преобразования камеры:")
    print(f"1. Перемещение: камера перемещается в начало координат")
    print(f"2. Поворот: камера поворачивается в стандартную ориентацию")
    print(f"3. Результат: сцена видна с точки зрения камеры")

def main():
    print("Задание 6: Реализация камеры")
    print("=" * 50)
    
    # Анализируем эффекты камеры
    analyze_camera_effects()
    
    # Демонстрируем обратное преобразование
    demonstrate_camera_inverse()
    
    # Исследуем свойства камеры
    investigate_camera_properties()
    
    print("\nЗадание 6 завершено!")
    print("Все изображения сохранены в папке images/task6/")

if __name__ == "__main__":
    main() 