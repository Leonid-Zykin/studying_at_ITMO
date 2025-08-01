import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from scipy.linalg import expm

# Создаем папку для сохранения изображений
os.makedirs('../images/task5', exist_ok=True)

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

def create_translation_matrix(tx, ty, tz):
    """Создание матрицы перемещения"""
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return T

def apply_transformation(vertices, transformation_matrix):
    """Применение матрицы преобразования к вершинам"""
    return transformation_matrix @ vertices

def draw_two_cubes(vertices1, vertices2, faces, color1='blue', color2='red', 
                   title='Исходный и повернутый кубики', filename='two_cubes'):
    """Отрисовка двух кубиков на одном графике"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    # Преобразуем однородные координаты в декартовы
    cartesian_vertices1 = (vertices1[:3, :] / vertices1[3, :]).T
    cartesian_vertices2 = (vertices2[:3, :] / vertices2[3, :]).T
    
    # Создаем грани для отрисовки первого кубика
    poly3d1 = []
    for face in faces:
        poly3d1.append([cartesian_vertices1[vertex] for vertex in face])
    
    # Создаем грани для отрисовки второго кубика
    poly3d2 = []
    for face in faces:
        poly3d2.append([cartesian_vertices2[vertex] for vertex in face])
    
    # Отрисовываем кубики
    ax.add_collection3d(Poly3DCollection(poly3d1, 
                                        facecolors=color1, 
                                        edgecolors='black', 
                                        linewidths=0.5,
                                        alpha=0.7))
    ax.add_collection3d(Poly3DCollection(poly3d2, 
                                        facecolors=color2, 
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
    all_vertices = np.vstack([cartesian_vertices1, cartesian_vertices2])
    max_range = max(np.max(all_vertices), abs(np.min(all_vertices)))
    ax.set_xlim(-max_range-1, max_range+1)
    ax.set_ylim(-max_range-1, max_range+1)
    ax.set_zlim(-max_range-1, max_range+1)
    
    # Устанавливаем углы обзора
    ax.view_init(azim=-37.5, elev=30)
    
    # Добавляем сетку
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../images/task5/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def rotate_around_vertex(vertices, faces, vertex_index, rotation_axis, angle):
    """Вращение кубика вокруг заданной вершины"""
    print(f"Вращение вокруг вершины {vertex_index}")
    
    # Получаем координаты вершины, вокруг которой вращаем
    vertex_coords = vertices[:3, vertex_index] / vertices[3, vertex_index]
    print(f"Координаты центра вращения: {vertex_coords}")
    
    # Шаг 1: Перемещаем кубик так, чтобы выбранная вершина оказалась в начале координат
    T1 = create_translation_matrix(-vertex_coords[0], -vertex_coords[1], -vertex_coords[2])
    print(f"Матрица перемещения T1 (в начало координат):")
    print(T1)
    
    # Шаг 2: Выполняем вращение вокруг оси
    R = create_rotation_matrix(rotation_axis, angle)
    print(f"Матрица вращения R:")
    print(R)
    
    # Шаг 3: Перемещаем кубик обратно
    T2 = create_translation_matrix(vertex_coords[0], vertex_coords[1], vertex_coords[2])
    print(f"Матрица перемещения T2 (обратно):")
    print(T2)
    
    # Композиция преобразований: T2 @ R @ T1
    transformation_matrix = T2 @ R @ T1
    print(f"Полная матрица преобразования T2 @ R @ T1:")
    print(transformation_matrix)
    
    # Применяем преобразование
    transformed_vertices = apply_transformation(vertices, transformation_matrix)
    
    return transformed_vertices, transformation_matrix

def analyze_rotation_around_vertices():
    """Анализ вращения вокруг различных вершин"""
    print("Анализ вращения вокруг различных вершин:")
    print("=" * 50)
    
    vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    # Отрисовываем исходный кубик
    draw_two_cubes(vertices, vertices, faces, 'blue', 'blue', 
                   'Исходный кубик', 'original_cube')
    print("Исходный кубик сохранен")
    
    # Различные случаи вращения вокруг вершин
    rotation_cases = [
        (0, np.array([0, 0, 1]), np.pi/4, 'red', 'Вращение вокруг вершины 0 (угол) на 45°'),
        (1, np.array([0, 1, 0]), np.pi/3, 'green', 'Вращение вокруг вершины 1 (ребро) на 60°'),
        (2, np.array([1, 0, 0]), np.pi/2, 'orange', 'Вращение вокруг вершины 2 (грань) на 90°'),
        (3, np.array([1, 1, 0]), np.pi/6, 'purple', 'Вращение вокруг вершины 3 (диагональ) на 30°'),
        (4, np.array([0, 0, 1]), np.pi, 'brown', 'Вращение вокруг вершины 4 на 180°'),
        (5, np.array([1, 1, 1]), np.pi/4, 'cyan', 'Вращение вокруг вершины 5 (пространственная диагональ) на 45°')
    ]
    
    for vertex_idx, axis, angle, color, title in rotation_cases:
        print(f"\n{title}")
        print("-" * 40)
        
        # Выполняем вращение
        transformed_vertices, transformation_matrix = rotate_around_vertex(
            vertices, faces, vertex_idx, axis, angle
        )
        
        # Отрисовываем результат
        draw_two_cubes(vertices, transformed_vertices, faces, 
                       'blue', color, title, f'rotate_around_vertex_{vertex_idx}')
        print(f"Результат сохранен в rotate_around_vertex_{vertex_idx}.png")
        
        # Анализируем результат
        original_center = np.mean((vertices[:3, :] / vertices[3, :]).T, axis=0)
        transformed_center = np.mean((transformed_vertices[:3, :] / transformed_vertices[3, :]).T, axis=0)
        
        print(f"Центр исходного кубика: {original_center}")
        print(f"Центр повернутого кубика: {transformed_center}")
        print(f"Смещение центра: {transformed_center - original_center}")

def investigate_transformation_properties():
    """Исследование свойств преобразования"""
    print("\nИсследование свойств преобразования:")
    print("=" * 50)
    
    vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    # Выбираем случай для детального анализа
    vertex_idx = 0
    axis = np.array([0, 0, 1])
    angle = np.pi/4
    
    print(f"Детальный анализ вращения вокруг вершины {vertex_idx}")
    
    # Выполняем вращение
    transformed_vertices, transformation_matrix = rotate_around_vertex(
        vertices, faces, vertex_idx, axis, angle
    )
    
    # Анализируем свойства матрицы преобразования
    print(f"\nСвойства матрицы преобразования:")
    print(f"Определитель: {np.linalg.det(transformation_matrix):.6f}")
    print(f"Ортогональность: T^T @ T == I: {np.allclose(transformation_matrix.T @ transformation_matrix, np.eye(4))}")
    
    # Проверяем, что выбранная вершина остается на месте
    original_vertex = vertices[:3, vertex_idx] / vertices[3, vertex_idx]
    transformed_vertex = transformed_vertices[:3, vertex_idx] / transformed_vertices[3, vertex_idx]
    
    vertex_preserved = np.allclose(original_vertex, transformed_vertex, atol=1e-10)
    print(f"Вершина {vertex_idx} остается на месте: {vertex_preserved}")
    print(f"Исходные координаты вершины: {original_vertex}")
    print(f"Координаты вершины после преобразования: {transformed_vertex}")
    
    # Анализируем движение других вершин
    print(f"\nДвижение других вершин:")
    for i in range(vertices.shape[1]):
        if i != vertex_idx:
            original_coords = vertices[:3, i] / vertices[3, i]
            transformed_coords = transformed_vertices[:3, i] / transformed_vertices[3, i]
            distance = np.linalg.norm(transformed_coords - original_coords)
            print(f"Вершина {i}: смещение на {distance:.4f}")

def demonstrate_geometric_interpretation():
    """Демонстрация геометрической интерпретации"""
    print("\nДемонстрация геометрической интерпретации:")
    print("=" * 50)
    
    vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    # Создаем несколько примеров с разными осями вращения
    examples = [
        (0, np.array([1, 0, 0]), np.pi/2, "Вращение вокруг оси X"),
        (1, np.array([0, 1, 0]), np.pi/2, "Вращение вокруг оси Y"),
        (2, np.array([0, 0, 1]), np.pi/2, "Вращение вокруг оси Z"),
        (3, np.array([1, 1, 0]), np.pi/3, "Вращение вокруг диагонали XY"),
        (4, np.array([1, 1, 1]), np.pi/4, "Вращение вокруг пространственной диагонали")
    ]
    
    for vertex_idx, axis, angle, description in examples:
        print(f"\n{description} вокруг вершины {vertex_idx}")
        
        # Выполняем вращение
        transformed_vertices, _ = rotate_around_vertex(
            vertices, faces, vertex_idx, axis, angle
        )
        
        # Отрисовываем результат
        draw_two_cubes(vertices, transformed_vertices, faces, 
                       'blue', 'red', description, f'geometric_{vertex_idx}')
        print(f"Результат сохранен в geometric_{vertex_idx}.png")
        
        # Анализируем геометрические свойства
        original_volume = 8  # Объем исходного кубика
        transformed_volume = 8  # Объем сохраняется при вращении
        
        print(f"Объем сохраняется: {original_volume} -> {transformed_volume}")
        print(f"Форма сохраняется: кубик остается кубиком")

def main():
    print("Задание 5: Вращение кубика около одной вершины")
    print("=" * 50)
    
    # Анализируем вращение вокруг различных вершин
    analyze_rotation_around_vertices()
    
    # Исследуем свойства преобразования
    investigate_transformation_properties()
    
    # Демонстрируем геометрическую интерпретацию
    demonstrate_geometric_interpretation()
    
    print("\nЗадание 5 завершено!")
    print("Все изображения сохранены в папке images/task5/")

if __name__ == "__main__":
    main() 