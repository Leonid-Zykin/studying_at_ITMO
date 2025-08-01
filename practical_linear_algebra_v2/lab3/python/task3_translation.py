import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# Создаем папку для сохранения изображений
os.makedirs('../images/task3', exist_ok=True)

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
    # Матрица перемещения в однородных координатах
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

def apply_transformation(vertices, transformation_matrix):
    """Применение матрицы преобразования к вершинам"""
    return transformation_matrix @ vertices

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
    plt.savefig(f'../images/task3/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_two_cubes(vertices1, vertices2, faces, color1='blue', color2='red', 
                   title='Два кубика', filename='two_cubes'):
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
    plt.savefig(f'../images/task3/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_translation_effects():
    """Анализ эффектов перемещения"""
    print("Анализ эффектов перемещения:")
    print("=" * 50)
    
    vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    # Отрисовываем исходный кубик
    draw_cube(vertices, faces, 'blue', 'Исходный кубик', 'original_cube')
    print("Исходный кубик сохранен")
    
    # Различные матрицы перемещения
    translation_cases = [
        (2, 0, 0, 'red', 'Перемещение по оси X на 2', 'translate_x'),
        (0, 2, 0, 'green', 'Перемещение по оси Y на 2', 'translate_y'),
        (0, 0, 2, 'orange', 'Перемещение по оси Z на 2', 'translate_z'),
        (1, 1, 0, 'purple', 'Перемещение по осям X и Y', 'translate_xy'),
        (1, 1, 1, 'brown', 'Перемещение по всем осям', 'translate_xyz'),
        (-1, 0, 1, 'cyan', 'Перемещение в противоположных направлениях', 'translate_mixed')
    ]
    
    for tx, ty, tz, color, title, filename in translation_cases:
        print(f"\nПеремещение: tx={tx}, ty={ty}, tz={tz}")
        
        # Создаем матрицу перемещения
        T = create_translation_matrix(tx, ty, tz)
        print(f"Матрица перемещения T:")
        print(T)
        
        # Применяем преобразование
        transformed_vertices = apply_transformation(vertices, T)
        
        # Отрисовываем результат
        draw_cube(transformed_vertices, faces, color, title, filename)
        print(f"Результат сохранен в {filename}.png")
    
    # Сравнительный анализ
    print("\nСравнительный анализ перемещения:")
    print("1. Перемещение не изменяет форму и размер объекта")
    print("2. Перемещение изменяет только положение центра объекта")
    print("3. Матрица перемещения имеет единичную подматрицу 3x3")
    print("4. Перемещение коммутативно: T1 @ T2 = T2 @ T1")

def investigate_ts_vs_st():
    """Исследование композиции TS vs ST"""
    print("\nИсследование композиции TS vs ST:")
    print("=" * 50)
    
    vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    # Создаем матрицы преобразований
    T = create_translation_matrix(2, 1, 0)  # Перемещение
    S = create_scaling_matrix(1.5, 1.5, 1.5)  # Масштабирование
    
    print("Матрица перемещения T:")
    print(T)
    print("\nМатрица масштабирования S:")
    print(S)
    
    # Применяем TS (сначала масштабирование, потом перемещение)
    vertices_ts = apply_transformation(apply_transformation(vertices, S), T)
    
    # Применяем ST (сначала перемещение, потом масштабирование)
    vertices_st = apply_transformation(apply_transformation(vertices, T), S)
    
    # Отрисовываем результаты
    draw_two_cubes(vertices_ts, vertices_st, faces, 
                   'red', 'blue', 
                   'TS (красный) vs ST (синий)', 'ts_vs_st')
    
    # Анализируем различия
    print("\nАнализ различий TS vs ST:")
    
    # Вычисляем центры кубиков
    cartesian_ts = (vertices_ts[:3, :] / vertices_ts[3, :]).T
    cartesian_st = (vertices_st[:3, :] / vertices_st[3, :]).T
    
    center_ts = np.mean(cartesian_ts, axis=0)
    center_st = np.mean(cartesian_st, axis=0)
    
    print(f"Центр кубика после TS: {center_ts}")
    print(f"Центр кубика после ST: {center_st}")
    
    # Вычисляем размеры кубиков
    size_ts = np.max(cartesian_ts, axis=0) - np.min(cartesian_ts, axis=0)
    size_st = np.max(cartesian_st, axis=0) - np.min(cartesian_st, axis=0)
    
    print(f"Размеры кубика после TS: {size_ts}")
    print(f"Размеры кубика после ST: {size_st}")
    
    # Проверяем эквивалентность
    ts_st_equivalent = np.allclose(vertices_ts, vertices_st)
    print(f"\nTS эквивалентно ST: {ts_st_equivalent}")
    
    if not ts_st_equivalent:
        print("Причины различий:")
        print("1. При TS: сначала масштабирование (изменение размера), потом перемещение")
        print("2. При ST: сначала перемещение, потом масштабирование (включая перемещение)")
        print("3. Масштабирование влияет на все координаты, включая перемещение")

def demonstrate_matrix_properties():
    """Демонстрация свойств матриц перемещения"""
    print("\nДемонстрация свойств матриц перемещения:")
    print("=" * 50)
    
    # Создаем различные матрицы перемещения
    T1 = create_translation_matrix(1, 0, 0)  # Перемещение по X
    T2 = create_translation_matrix(0, 1, 0)  # Перемещение по Y
    T3 = create_translation_matrix(0, 0, 1)  # Перемещение по Z
    
    print("Матрица T1 (перемещение по X):")
    print(T1)
    print(f"Определитель: {np.linalg.det(T1)}")
    
    print("\nМатрица T2 (перемещение по Y):")
    print(T2)
    print(f"Определитель: {np.linalg.det(T2)}")
    
    print("\nМатрица T3 (перемещение по Z):")
    print(T3)
    print(f"Определитель: {np.linalg.det(T3)}")
    
    # Композиция преобразований
    T_combined = T1 @ T2 @ T3
    print("\nКомпозиция T1 @ T2 @ T3:")
    print(T_combined)
    print(f"Определитель: {np.linalg.det(T_combined)}")
    
    # Проверяем коммутативность
    T1_T2 = T1 @ T2
    T2_T1 = T2 @ T1
    print(f"\nКоммутативность: T1 @ T2 == T2 @ T1: {np.allclose(T1_T2, T2_T1)}")
    
    # Обратная матрица
    T1_inv = np.linalg.inv(T1)
    print(f"\nОбратная матрица T1^(-1):")
    print(T1_inv)
    print(f"Проверка: T1 @ T1^(-1) == I: {np.allclose(T1 @ T1_inv, np.eye(4))}")

def main():
    print("Задание 3: Перемещение кубика")
    print("=" * 50)
    
    # Анализируем эффекты перемещения
    analyze_translation_effects()
    
    # Исследуем композицию TS vs ST
    investigate_ts_vs_st()
    
    # Демонстрируем свойства матриц
    demonstrate_matrix_properties()
    
    print("\nЗадание 3 завершено!")
    print("Все изображения сохранены в папке images/task3/")

if __name__ == "__main__":
    main() 