import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# Создаем папку для сохранения изображений
os.makedirs('../images/task2', exist_ok=True)

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

def create_scaling_matrix(sx, sy, sz):
    """Создание матрицы масштабирования"""
    # Матрица масштабирования в однородных координатах
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
    ax.set_xlim(-max_range-0.5, max_range+0.5)
    ax.set_ylim(-max_range-0.5, max_range+0.5)
    ax.set_zlim(-max_range-0.5, max_range+0.5)
    
    # Устанавливаем углы обзора
    ax.view_init(azim=-37.5, elev=30)
    
    # Добавляем сетку
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../images/task2/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_scaling_effects():
    """Анализ эффектов масштабирования"""
    print("Анализ эффектов масштабирования:")
    print("=" * 50)
    
    vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    # Отрисовываем исходный кубик
    draw_cube(vertices, faces, 'blue', 'Исходный кубик', 'original_cube')
    print("Исходный кубик сохранен")
    
    # Различные матрицы масштабирования
    scaling_cases = [
        (2, 1, 1, 'red', 'Растяжение по оси X (2x)', 'scale_x'),
        (1, 2, 1, 'green', 'Растяжение по оси Y (2x)', 'scale_y'),
        (1, 1, 2, 'orange', 'Растяжение по оси Z (2x)', 'scale_z'),
        (2, 2, 1, 'purple', 'Растяжение по осям X и Y (2x)', 'scale_xy'),
        (0.5, 0.5, 0.5, 'brown', 'Сжатие по всем осям (0.5x)', 'scale_all_small'),
        (2, 2, 2, 'cyan', 'Растяжение по всем осям (2x)', 'scale_all_large'),
        (1, 0.5, 2, 'magenta', 'Комбинированное масштабирование', 'scale_combined')
    ]
    
    for sx, sy, sz, color, title, filename in scaling_cases:
        print(f"\nМасштабирование: sx={sx}, sy={sy}, sz={sz}")
        
        # Создаем матрицу масштабирования
        S = create_scaling_matrix(sx, sy, sz)
        print(f"Матрица масштабирования S:")
        print(S)
        
        # Применяем преобразование
        transformed_vertices = apply_transformation(vertices, S)
        
        # Анализируем результат
        original_volume = 8  # Объем исходного кубика (2x2x2)
        new_volume = sx * sy * sz * original_volume
        print(f"Объем кубика изменился с {original_volume} до {new_volume}")
        
        # Отрисовываем результат
        draw_cube(transformed_vertices, faces, color, title, filename)
        print(f"Результат сохранен в {filename}.png")
    
    # Сравнительный анализ
    print("\nСравнительный анализ масштабирования:")
    print("1. Масштабирование по одной оси: кубик становится прямоугольным параллелепипедом")
    print("2. Масштабирование по двум осям: кубик становится призмой")
    print("3. Равномерное масштабирование: сохраняется форма, меняется размер")
    print("4. Объем изменяется пропорционально произведению коэффициентов масштабирования")

def demonstrate_matrix_properties():
    """Демонстрация свойств матриц масштабирования"""
    print("\nДемонстрация свойств матриц масштабирования:")
    print("=" * 50)
    
    # Создаем различные матрицы масштабирования
    S1 = create_scaling_matrix(2, 1, 1)  # Растяжение по X
    S2 = create_scaling_matrix(1, 2, 1)  # Растяжение по Y
    S3 = create_scaling_matrix(1, 1, 2)  # Растяжение по Z
    
    print("Матрица S1 (растяжение по X):")
    print(S1)
    print(f"Определитель: {np.linalg.det(S1)}")
    
    print("\nМатрица S2 (растяжение по Y):")
    print(S2)
    print(f"Определитель: {np.linalg.det(S2)}")
    
    print("\nМатрица S3 (растяжение по Z):")
    print(S3)
    print(f"Определитель: {np.linalg.det(S3)}")
    
    # Композиция преобразований
    S_combined = S1 @ S2 @ S3
    print("\nКомпозиция S1 @ S2 @ S3:")
    print(S_combined)
    print(f"Определитель: {np.linalg.det(S_combined)}")
    
    # Проверяем коммутативность
    S1_S2 = S1 @ S2
    S2_S1 = S2 @ S1
    print(f"\nКоммутативность: S1 @ S2 == S2 @ S1: {np.allclose(S1_S2, S2_S1)}")
    
    # Обратная матрица
    S1_inv = np.linalg.inv(S1)
    print(f"\nОбратная матрица S1^(-1):")
    print(S1_inv)
    print(f"Проверка: S1 @ S1^(-1) == I: {np.allclose(S1 @ S1_inv, np.eye(4))}")

def main():
    print("Задание 2: Масштабирование кубика")
    print("=" * 50)
    
    # Анализируем эффекты масштабирования
    analyze_scaling_effects()
    
    # Демонстрируем свойства матриц
    demonstrate_matrix_properties()
    
    print("\nЗадание 2 завершено!")
    print("Все изображения сохранены в папке images/task2/")

if __name__ == "__main__":
    main() 