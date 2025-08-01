import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from scipy.linalg import expm

# Создаем папку для сохранения изображений
os.makedirs('../images/task8', exist_ok=True)

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

def create_scaling_matrix(sx, sy, sz):
    """Создание матрицы масштабирования"""
    S = np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])
    return S

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

def create_house():
    """Создание домика из блоков"""
    print("Создание домика из блоков:")
    print("=" * 50)
    
    # Создаем базовый кубик
    base_vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    # Создаем части домика
    house_objects = []
    
    # Фундамент (основание дома)
    S_foundation = create_scaling_matrix(4, 3, 0.5)
    T_foundation = create_translation_matrix(0, 0, 0)
    foundation = {
        'vertices': apply_transformation(apply_transformation(base_vertices, S_foundation), T_foundation),
        'color': 'brown',
        'name': 'Фундамент'
    }
    house_objects.append(foundation)
    
    # Стены дома
    # Передняя стена
    S_wall_front = create_scaling_matrix(3, 0.2, 2)
    T_wall_front = create_translation_matrix(0, 1.4, 1)
    wall_front = {
        'vertices': apply_transformation(apply_transformation(base_vertices, S_wall_front), T_wall_front),
        'color': 'lightblue',
        'name': 'Передняя стена'
    }
    house_objects.append(wall_front)
    
    # Задняя стена
    S_wall_back = create_scaling_matrix(3, 0.2, 2)
    T_wall_back = create_translation_matrix(0, -1.4, 1)
    wall_back = {
        'vertices': apply_transformation(apply_transformation(base_vertices, S_wall_back), T_wall_back),
        'color': 'lightblue',
        'name': 'Задняя стена'
    }
    house_objects.append(wall_back)
    
    # Левая стена
    S_wall_left = create_scaling_matrix(0.2, 3, 2)
    T_wall_left = create_translation_matrix(-1.4, 0, 1)
    wall_left = {
        'vertices': apply_transformation(apply_transformation(base_vertices, S_wall_left), T_wall_left),
        'color': 'lightblue',
        'name': 'Левая стена'
    }
    house_objects.append(wall_left)
    
    # Правая стена
    S_wall_right = create_scaling_matrix(0.2, 3, 2)
    T_wall_right = create_translation_matrix(1.4, 0, 1)
    wall_right = {
        'vertices': apply_transformation(apply_transformation(base_vertices, S_wall_right), T_wall_right),
        'color': 'lightblue',
        'name': 'Правая стена'
    }
    house_objects.append(wall_right)
    
    # Крыша (треугольная призма)
    # Левая часть крыши
    S_roof_left = create_scaling_matrix(2, 3, 0.2)
    R_roof_left = create_rotation_matrix(np.array([0, 1, 0]), np.pi/6)
    T_roof_left = create_translation_matrix(-1, 0, 2.5)
    roof_left = {
        'vertices': apply_transformation(apply_transformation(apply_transformation(base_vertices, R_roof_left), S_roof_left), T_roof_left),
        'color': 'red',
        'name': 'Левая часть крыши'
    }
    house_objects.append(roof_left)
    
    # Правая часть крыши
    S_roof_right = create_scaling_matrix(2, 3, 0.2)
    R_roof_right = create_rotation_matrix(np.array([0, 1, 0]), -np.pi/6)
    T_roof_right = create_translation_matrix(1, 0, 2.5)
    roof_right = {
        'vertices': apply_transformation(apply_transformation(apply_transformation(base_vertices, R_roof_right), S_roof_right), T_roof_right),
        'color': 'red',
        'name': 'Правая часть крыши'
    }
    house_objects.append(roof_right)
    
    # Дверь
    S_door = create_scaling_matrix(0.8, 0.1, 1.5)
    T_door = create_translation_matrix(0, 1.5, 0.75)
    door = {
        'vertices': apply_transformation(apply_transformation(base_vertices, S_door), T_door),
        'color': 'darkgreen',
        'name': 'Дверь'
    }
    house_objects.append(door)
    
    # Окна
    # Переднее окно
    S_window_front = create_scaling_matrix(0.8, 0.1, 0.8)
    T_window_front = create_translation_matrix(0, 1.5, 1.5)
    window_front = {
        'vertices': apply_transformation(apply_transformation(base_vertices, S_window_front), T_window_front),
        'color': 'yellow',
        'name': 'Переднее окно'
    }
    house_objects.append(window_front)
    
    # Заднее окно
    S_window_back = create_scaling_matrix(0.8, 0.1, 0.8)
    T_window_back = create_translation_matrix(0, -1.5, 1.5)
    window_back = {
        'vertices': apply_transformation(apply_transformation(base_vertices, S_window_back), T_window_back),
        'color': 'yellow',
        'name': 'Заднее окно'
    }
    house_objects.append(window_back)
    
    # Дымоход
    S_chimney = create_scaling_matrix(0.3, 0.3, 1)
    T_chimney = create_translation_matrix(0.5, 0, 3.5)
    chimney = {
        'vertices': apply_transformation(apply_transformation(base_vertices, S_chimney), T_chimney),
        'color': 'gray',
        'name': 'Дымоход'
    }
    house_objects.append(chimney)
    
    # Труба дымохода
    S_pipe = create_scaling_matrix(0.2, 0.2, 0.5)
    T_pipe = create_translation_matrix(0.5, 0, 4.5)
    pipe = {
        'vertices': apply_transformation(apply_transformation(base_vertices, S_pipe), T_pipe),
        'color': 'darkgray',
        'name': 'Труба'
    }
    house_objects.append(pipe)
    
    return house_objects, faces

def draw_house(house_objects, faces, title='Домик', filename='house'):
    """Отрисовка домика"""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    # Отрисовываем каждый элемент домика
    for i, obj in enumerate(house_objects):
        vertices = obj['vertices']
        color = obj['color']
        
        # Преобразуем однородные координаты в декартовы
        cartesian_vertices = (vertices[:3, :] / vertices[3, :]).T
        
        # Создаем грани для отрисовки
        poly3d = []
        for face in faces:
            poly3d.append([cartesian_vertices[vertex] for vertex in face])
        
        # Отрисовываем элемент
        ax.add_collection3d(Poly3DCollection(poly3d, 
                                            facecolors=color, 
                                            edgecolors='black', 
                                            linewidths=0.5,
                                            alpha=0.8))
    
    # Настройка осей
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Устанавливаем одинаковые масштабы по осям
    ax.set_box_aspect([1, 1, 1])
    
    # Устанавливаем пределы осей
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 6)
    
    # Устанавливаем углы обзора
    ax.view_init(azim=-37.5, elev=30)
    
    # Добавляем сетку
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../images/task8/{filename}.png', dpi=300, bbox_inches='tight')
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
    
    return C

def create_perspective_matrix(near, far, fov, aspect_ratio):
    """Создание матрицы перспективы"""
    # Вычисляем параметры перспективы
    f = 1.0 / np.tan(fov / 2)
    
    # Матрица перспективы
    P = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])
    
    return P

def apply_camera_and_perspective(house_objects, camera_matrix, perspective_matrix):
    """Применение камеры и перспективы к домику"""
    transformed_objects = []
    
    for obj in house_objects:
        # Применяем камеру
        camera_transformed = apply_transformation(obj['vertices'], camera_matrix)
        # Применяем перспективу
        final_transformed = apply_transformation(camera_transformed, perspective_matrix)
        
        transformed_obj = {
            'vertices': final_transformed,
            'color': obj['color'],
            'name': obj['name']
        }
        transformed_objects.append(transformed_obj)
    
    return transformed_objects

def analyze_house_from_different_angles():
    """Анализ домика с разных углов обзора"""
    print("Анализ домика с разных углов обзора:")
    print("=" * 50)
    
    # Создаем домик
    house_objects, faces = create_house()
    
    # Отрисовываем исходный домик
    draw_house(house_objects, faces, 'Домик', 'house_original')
    print("Исходный домик сохранен")
    
    # Различные позиции камеры
    camera_positions = [
        (np.array([5, 5, 3]), np.array([0, 0, 2]), np.array([0, 0, 1]), 
         'Вид сбоку и сверху', 'house_side_top'),
        (np.array([0, 0, 8]), np.array([0, 0, 2]), np.array([0, 1, 0]), 
         'Вид спереди', 'house_front'),
        (np.array([8, 0, 0]), np.array([0, 0, 2]), np.array([0, 0, 1]), 
         'Вид справа', 'house_right'),
        (np.array([3, 3, 2]), np.array([0, 0, 2]), np.array([0, 0, 1]), 
         'Вид под углом', 'house_angle')
    ]
    
    for camera_pos, camera_target, up_vector, title, filename in camera_positions:
        print(f"\n{title}")
        print(f"Позиция камеры: {camera_pos}")
        
        # Создаем матрицу камеры
        C = create_camera_matrix(camera_pos, camera_target, up_vector)
        
        # Создаем матрицу перспективы
        P = create_perspective_matrix(0.1, 10.0, np.pi/4, 1.0)
        
        # Применяем преобразования
        transformed_objects = apply_camera_and_perspective(house_objects, C, P)
        
        # Отрисовываем результат
        draw_house(transformed_objects, faces, f'Домик: {title}', filename)
        print(f"Результат сохранен в {filename}.png")

def demonstrate_house_construction():
    """Демонстрация построения домика"""
    print("\nДемонстрация построения домика:")
    print("=" * 50)
    
    # Создаем базовый кубик
    base_vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    # Показываем этапы построения
    construction_stages = [
        ("Фундамент", create_scaling_matrix(4, 3, 0.5), create_translation_matrix(0, 0, 0), 'brown'),
        ("Стены", create_scaling_matrix(3, 0.2, 2), create_translation_matrix(0, 1.4, 1), 'lightblue'),
        ("Крыша", create_scaling_matrix(2, 3, 0.2), create_translation_matrix(0, 0, 2.5), 'red'),
        ("Дверь", create_scaling_matrix(0.8, 0.1, 1.5), create_translation_matrix(0, 1.5, 0.75), 'darkgreen'),
        ("Окна", create_scaling_matrix(0.8, 0.1, 0.8), create_translation_matrix(0, 1.5, 1.5), 'yellow'),
        ("Дымоход", create_scaling_matrix(0.3, 0.3, 1), create_translation_matrix(0.5, 0, 3.5), 'gray')
    ]
    
    for stage_name, S, T, color in construction_stages:
        print(f"Добавляем {stage_name}")
        
        # Создаем элемент
        element_vertices = apply_transformation(apply_transformation(base_vertices, S), T)
        element = {
            'vertices': element_vertices,
            'color': color,
            'name': stage_name
        }
        
        # Отрисовываем текущий этап
        draw_house([element], faces, f'Этап: {stage_name}', f'construction_{stage_name.lower()}')
        print(f"Этап сохранен в construction_{stage_name.lower()}.png")

def investigate_house_properties():
    """Исследование свойств домика"""
    print("\nИсследование свойств домика:")
    print("=" * 50)
    
    # Создаем домик
    house_objects, faces = create_house()
    
    print("Статистика домика:")
    print(f"Количество элементов: {len(house_objects)}")
    
    # Анализируем каждый элемент
    for i, obj in enumerate(house_objects):
        vertices = obj['vertices']
        name = obj['name']
        
        # Вычисляем размеры элемента
        cartesian_vertices = (vertices[:3, :] / vertices[3, :]).T
        size = np.max(cartesian_vertices, axis=0) - np.min(cartesian_vertices, axis=0)
        volume = np.prod(size)
        
        print(f"{name}: размеры {size}, объем {volume:.2f}")
    
    # Общий анализ
    print(f"\nОбщий анализ:")
    print(f"1. Домик построен из {len(house_objects)} кубических блоков")
    print(f"2. Использованы матрицы масштабирования, перемещения и вращения")
    print(f"3. Каждый элемент имеет свой цвет и назначение")
    print(f"4. Структура домика: фундамент -> стены -> крыша -> детали")

def main():
    print("Задание 8: Построение домика (почти Blender)")
    print("=" * 50)
    
    # Анализируем домик с разных углов
    analyze_house_from_different_angles()
    
    # Демонстрируем построение
    demonstrate_house_construction()
    
    # Исследуем свойства
    investigate_house_properties()
    
    print("\nЗадание 8 завершено!")
    print("Все изображения сохранены в папке images/task8/")

if __name__ == "__main__":
    main() 