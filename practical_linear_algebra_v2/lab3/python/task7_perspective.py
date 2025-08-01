import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from scipy.linalg import expm

# Создаем папку для сохранения изображений
os.makedirs('../images/task7', exist_ok=True)

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
    plt.savefig(f'../images/task7/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_perspective_scene(scene_objects, faces, title='Сцена с перспективой', filename='perspective_scene'):
    """Отрисовка сцены с перспективным преобразованием"""
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
    
    # Динамически устанавливаем пределы осей
    all_vertices = []
    for obj in scene_objects:
        cartesian_vertices = (obj['vertices'][:3, :] / obj['vertices'][3, :]).T
        all_vertices.extend(cartesian_vertices)
    
    all_vertices = np.array(all_vertices)
    max_range = max(np.max(all_vertices), abs(np.min(all_vertices)))
    ax.set_xlim(-max_range-1, max_range+1)
    ax.set_ylim(-max_range-1, max_range+1)
    ax.set_zlim(-max_range-1, max_range+1)
    
    # Устанавливаем углы обзора
    ax.view_init(azim=-37.5, elev=30)
    
    # Добавляем сетку
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../images/task7/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_perspective_effects():
    """Анализ эффектов перспективы"""
    print("Анализ эффектов перспективы:")
    print("=" * 50)
    
    # Создаем сцену
    scene_objects, faces = create_scene()
    
    # Отрисовываем исходную сцену
    draw_scene(scene_objects, faces, 'Исходная сцена', 'original_scene')
    print("Исходная сцена сохранена")
    
    # Различные параметры перспективы
    perspective_cases = [
        (0.1, 10.0, np.pi/4, 1.0, 'Стандартная перспектива', 'perspective_standard'),
        (0.1, 10.0, np.pi/6, 1.0, 'Узкий угол обзора', 'perspective_narrow'),
        (0.1, 10.0, np.pi/3, 1.0, 'Широкий угол обзора', 'perspective_wide'),
        (0.1, 5.0, np.pi/4, 1.0, 'Близкая дальняя плоскость', 'perspective_close_far'),
        (0.5, 10.0, np.pi/4, 1.0, 'Дальняя ближняя плоскость', 'perspective_far_near'),
        (0.1, 10.0, np.pi/4, 2.0, 'Широкое соотношение сторон', 'perspective_wide_aspect'),
        (0.1, 10.0, np.pi/4, 0.5, 'Узкое соотношение сторон', 'perspective_narrow_aspect')
    ]
    
    for near, far, fov, aspect_ratio, title, filename in perspective_cases:
        print(f"\n{title}")
        print(f"Ближняя плоскость: {near}")
        print(f"Дальняя плоскость: {far}")
        print(f"Угол обзора: {np.degrees(fov):.1f}°")
        print(f"Соотношение сторон: {aspect_ratio}")
        
        # Создаем матрицу перспективы
        P = create_perspective_matrix(near, far, fov, aspect_ratio)
        print(f"Матрица перспективы P:")
        print(P)
        
        # Применяем перспективное преобразование ко всем объектам
        transformed_objects = []
        for obj in scene_objects:
            transformed_vertices = apply_transformation(obj['vertices'], P)
            transformed_obj = {
                'vertices': transformed_vertices,
                'color': obj['color'],
                'name': obj['name']
            }
            transformed_objects.append(transformed_obj)
        
        # Отрисовываем результат
        draw_perspective_scene(transformed_objects, faces, f'Сцена: {title}', filename)
        print(f"Результат сохранен в {filename}.png")
        
        # Анализируем эффект
        print(f"Эффект: объекты деформируются в зависимости от расстояния")
        print(f"Ближние объекты увеличиваются, дальние уменьшаются")

def investigate_perspective_properties():
    """Исследование свойств матрицы перспективы"""
    print("\nИсследование свойств матрицы перспективы:")
    print("=" * 50)
    
    # Создаем матрицу перспективы
    near, far, fov, aspect_ratio = 0.1, 10.0, np.pi/4, 1.0
    P = create_perspective_matrix(near, far, fov, aspect_ratio)
    
    print("Свойства матрицы перспективы:")
    print(f"Определитель P: {np.linalg.det(P):.6f}")
    print(f"Определитель не равен 1: {not np.allclose(np.linalg.det(P), 1)}")
    
    # Анализируем структуру матрицы
    print(f"\nСтруктура матрицы перспективы:")
    print(f"P[3,3] = 0 (не равно 1): {P[3,3] == 0}")
    print(f"P[3,2] = -1 (отрицательное значение): {P[3,2] == -1}")
    
    # Проверяем эффект на однородные координаты
    print(f"\nЭффект на однородные координаты:")
    test_point = np.array([1, 1, 5, 1])  # Точка в пространстве
    transformed_point = P @ test_point
    print(f"Исходная точка: {test_point}")
    print(f"После перспективы: {transformed_point}")
    print(f"W координата изменилась: {transformed_point[3] != 1}")
    
    # Анализируем деформацию пространства
    print(f"\nДеформация пространства:")
    print(f"1. Ближние объекты (z ≈ near): увеличиваются")
    print(f"2. Дальние объекты (z ≈ far): уменьшаются")
    print(f"3. Объекты на бесконечности (z → ∞): стремятся к нулю")
    print(f"4. W координата становится функцией от Z")

def demonstrate_perspective_interpretation():
    """Демонстрация интерпретации перспективы"""
    print("\nДемонстрация интерпретации перспективы:")
    print("=" * 50)
    
    # Создаем простую сцену с одним кубиком
    base_vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    # Перемещаем кубик дальше
    T = create_translation_matrix(0, 0, 5)
    cube_vertices = apply_transformation(base_vertices, T)
    
    # Создаем матрицу перспективы
    P = create_perspective_matrix(0.1, 10.0, np.pi/4, 1.0)
    
    # Применяем перспективу
    perspective_vertices = apply_transformation(cube_vertices, P)
    
    # Анализируем результат
    print("Анализ перспективного преобразования:")
    
    # Исходные координаты
    original_coords = (cube_vertices[:3, :] / cube_vertices[3, :]).T
    print(f"Исходные координаты кубика:")
    for i, coord in enumerate(original_coords):
        print(f"  Вершина {i}: {coord}")
    
    # Координаты после перспективы
    perspective_coords = (perspective_vertices[:3, :] / perspective_vertices[3, :]).T
    print(f"\nКоординаты после перспективы:")
    for i, coord in enumerate(perspective_coords):
        print(f"  Вершина {i}: {coord}")
    
    # Анализируем масштабирование
    print(f"\nАнализ масштабирования:")
    original_size = np.max(original_coords) - np.min(original_coords)
    perspective_size = np.max(perspective_coords) - np.min(perspective_coords)
    scale_factor = perspective_size / original_size
    print(f"Исходный размер: {original_size:.4f}")
    print(f"Размер после перспективы: {perspective_size:.4f}")
    print(f"Коэффициент масштабирования: {scale_factor:.4f}")
    
    # Отрисовываем результаты
    original_scene = [{'vertices': cube_vertices, 'color': 'blue', 'name': 'Кубик'}]
    perspective_scene = [{'vertices': perspective_vertices, 'color': 'red', 'name': 'Кубик с перспективой'}]
    
    draw_scene(original_scene, faces, 'Исходный кубик', 'original_cube_for_perspective')
    draw_perspective_scene(perspective_scene, faces, 'Кубик с перспективой', 'perspective_cube')

def main():
    print("Задание 7: Реализация перспективы")
    print("=" * 50)
    
    # Анализируем эффекты перспективы
    analyze_perspective_effects()
    
    # Исследуем свойства перспективы
    investigate_perspective_properties()
    
    # Демонстрируем интерпретацию
    demonstrate_perspective_interpretation()
    
    print("\nЗадание 7 завершено!")
    print("Все изображения сохранены в папке images/task7/")

if __name__ == "__main__":
    main() 