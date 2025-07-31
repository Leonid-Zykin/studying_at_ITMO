import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import math

# Выбранные числа
a = 3
b = 5
c = 7
d = 11

# Исходный многоугольник
vertices = np.array([
    [0, 0],    # вершина 1
    [2, 1],    # вершина 2
    [3, 3],    # вершина 3
    [2, 4],    # вершина 4
    [0, 3],    # вершина 5
    [-1, 1]    # вершина 6
])

def apply_transformation(vertices, matrix, title, filename, show_eigenvectors=False, eigenvals=None, eigenvecs=None):
    """Применяет матричное преобразование к вершинам многоугольника и сохраняет результат"""
    # Применение преобразования
    transformed_vertices = np.dot(vertices, matrix.T)
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Отрисовка исходного многоугольника (пунктиром)
    polygon_original = Polygon(vertices, facecolor='none', edgecolor='gray', 
                              linewidth=1, linestyle='--', alpha=0.5)
    ax.add_patch(polygon_original)
    
    # Отрисовка преобразованного многоугольника
    polygon_transformed = Polygon(transformed_vertices, facecolor='lightgreen', 
                                 edgecolor='green', linewidth=2)
    ax.add_patch(polygon_transformed)
    
    # Отрисовка вершин
    ax.scatter(vertices[:, 0], vertices[:, 1], color='red', s=100, zorder=5, alpha=0.5)
    ax.scatter(transformed_vertices[:, 0], transformed_vertices[:, 1], color='blue', s=100, zorder=5)
    
    # Отрисовка собственных векторов
    if show_eigenvectors and eigenvals is not None and eigenvecs is not None:
        for i, (eigenval, eigenvec) in enumerate(zip(eigenvals, eigenvecs.T)):
            # Нормализация собственного вектора для отображения
            eigenvec_norm = eigenvec / np.linalg.norm(eigenvec) * 3
            ax.quiver(0, 0, eigenvec_norm[0], eigenvec_norm[1], 
                     color=['red', 'blue'][i], linewidth=2, 
                     label=f'λ={eigenval:.2f}')
            ax.legend()
    
    # Настройка осей
    all_x = np.concatenate([vertices[:, 0], transformed_vertices[:, 0]])
    all_y = np.concatenate([vertices[:, 1], transformed_vertices[:, 1]])
    
    margin = 1
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    
    # Добавление координатных осей
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'images/task1/{filename}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return transformed_vertices

def get_eigenvalues_eigenvectors(matrix):
    """Вычисляет собственные значения и векторы матрицы"""
    eigenvals, eigenvecs = np.linalg.eig(matrix)
    return eigenvals, eigenvecs

# Задание 1: Создание различных матричных преобразований

print("=== ЗАДАНИЕ 1: Создание матричных преобразований ===")

# 1. Отражение относительно прямой y = ax
angle = math.atan(a)
reflection_matrix = np.array([
    [math.cos(2*angle), math.sin(2*angle)],
    [math.sin(2*angle), -math.cos(2*angle)]
])
print(f"1. Матрица отражения относительно y={a}x:\n{reflection_matrix}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(reflection_matrix)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, reflection_matrix, 
                   f'Отражение относительно прямой y={a}x', 
                   'reflection_y_ax', True, eigenvals, eigenvecs)

# 2. Отображение всей плоскости в прямую y = bx
projection_matrix = np.array([
    [0, 0],
    [b, 0]
])
print(f"\n2. Матрица проекции на y={b}x:\n{projection_matrix}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(projection_matrix)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, projection_matrix, 
                   f'Проекция на прямую y={b}x', 
                   'projection_y_bx', True, eigenvals, eigenvecs)

# 3. Поворот на 10c градусов против часовой стрелки
rotation_angle = math.radians(10*c)
rotation_matrix = np.array([
    [math.cos(rotation_angle), -math.sin(rotation_angle)],
    [math.sin(rotation_angle), math.cos(rotation_angle)]
])
print(f"\n3. Матрица поворота на {10*c}°:\n{rotation_matrix}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(rotation_matrix)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, rotation_matrix, 
                   f'Поворот на {10*c}° против часовой стрелки', 
                   'rotation_10c_degrees', True, eigenvals, eigenvecs)

# 4. Центральная симметрия
central_symmetry_matrix = np.array([
    [-1, 0],
    [0, -1]
])
print(f"\n4. Матрица центральной симметрии:\n{central_symmetry_matrix}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(central_symmetry_matrix)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, central_symmetry_matrix, 
                   'Центральная симметрия относительно начала координат', 
                   'central_symmetry', True, eigenvals, eigenvecs)

# 5. Отражение + поворот на 10d градусов по часовой стрелке
rotation_angle_d = math.radians(-10*d)  # по часовой стрелке
rotation_matrix_d = np.array([
    [math.cos(rotation_angle_d), -math.sin(rotation_angle_d)],
    [math.sin(rotation_angle_d), math.cos(rotation_angle_d)]
])
combined_matrix = np.dot(rotation_matrix_d, reflection_matrix)
print(f"\n5. Матрица отражение + поворот на {10*d}° по часовой стрелке:\n{combined_matrix}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(combined_matrix)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, combined_matrix, 
                   f'Отражение + поворот на {10*d}° по часовой стрелке', 
                   'reflection_plus_rotation', True, eigenvals, eigenvecs)

# 6. Отображение, переводящее y=0 в y=ax и x=0 в y=bx
# Это матрица, которая переводит базисные векторы
matrix_6 = np.array([
    [1, a],
    [0, b]
])
print(f"\n6. Матрица перевода y=0→y={a}x, x=0→y={b}x:\n{matrix_6}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(matrix_6)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, matrix_6, 
                   f'Перевод y=0→y={a}x, x=0→y={b}x', 
                   'transformation_6', True, eigenvals, eigenvecs)

# 7. Отображение, переводящее y=ax в y=0 и y=bx в x=0
# Обратная матрица к предыдущей
matrix_7 = np.linalg.inv(matrix_6)
print(f"\n7. Матрица перевода y={a}x→y=0, y={b}x→x=0:\n{matrix_7}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(matrix_7)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, matrix_7, 
                   f'Перевод y={a}x→y=0, y={b}x→x=0', 
                   'transformation_7', True, eigenvals, eigenvecs)

# 8. Отображение, меняющее местами прямые y=ax и y=bx
# Это матрица перестановки с масштабированием
matrix_8 = np.array([
    [b/a, 0],
    [0, a/b]
])
print(f"\n8. Матрица обмена y={a}x↔y={b}x:\n{matrix_8}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(matrix_8)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, matrix_8, 
                   f'Обмен прямых y={a}x↔y={b}x', 
                   'transformation_8', True, eigenvals, eigenvecs)

# 9. Отображение, переводящее круг единичной площади в круг площади c
# Масштабирование с коэффициентом sqrt(c)
scale_factor = math.sqrt(c)
matrix_9 = np.array([
    [scale_factor, 0],
    [0, scale_factor]
])
print(f"\n9. Матрица масштабирования в {scale_factor} раз (площадь {c}):\n{matrix_9}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(matrix_9)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, matrix_9, 
                   f'Масштабирование в {scale_factor} раз (площадь {c})', 
                   'scaling_c', True, eigenvals, eigenvecs)

# 10. Отображение, переводящее круг единичной площади в некруг площади d
# Эллиптическое преобразование
matrix_10 = np.array([
    [math.sqrt(d), 0],
    [0, 1/math.sqrt(d)]
])
print(f"\n10. Матрица эллиптического преобразования (площадь {d}):\n{matrix_10}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(matrix_10)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, matrix_10, 
                   f'Эллиптическое преобразование (площадь {d})', 
                   'elliptic_d', True, eigenvals, eigenvecs)

# 11. Отображение с перпендикулярными собственными векторами
# Симметричная матрица
matrix_11 = np.array([
    [2, 1],
    [1, 3]
])
print(f"\n11. Матрица с перпендикулярными собственными векторами:\n{matrix_11}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(matrix_11)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, matrix_11, 
                   'Отображение с перпендикулярными собственными векторами', 
                   'perpendicular_eigenvectors', True, eigenvals, eigenvecs)

# 12. Отображение без двух неколлинеарных собственных векторов
# Матрица с кратным собственным значением
matrix_12 = np.array([
    [2, 1],
    [0, 2]
])
print(f"\n12. Матрица без двух неколлинеарных собственных векторов:\n{matrix_12}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(matrix_12)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, matrix_12, 
                   'Отображение без двух неколлинеарных собственных векторов', 
                   'single_eigenvector', True, eigenvals, eigenvecs)

# 13. Отображение без вещественных собственных векторов
# Матрица с комплексными собственными значениями
matrix_13 = np.array([
    [0, -1],
    [1, 0]
])
print(f"\n13. Матрица без вещественных собственных векторов:\n{matrix_13}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(matrix_13)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, matrix_13, 
                   'Отображение без вещественных собственных векторов', 
                   'complex_eigenvalues', True, eigenvals, eigenvecs)

# 14. Отображение, для которого любой ненулевой вектор является собственным
# Скалярная матрица
matrix_14 = np.array([
    [2, 0],
    [0, 2]
])
print(f"\n14. Матрица, для которой любой вектор собственный:\n{matrix_14}")
eigenvals, eigenvecs = get_eigenvalues_eigenvectors(matrix_14)
print(f"Собственные значения: {eigenvals}")
print(f"Собственные векторы:\n{eigenvecs}")
apply_transformation(vertices, matrix_14, 
                   'Отображение с любым вектором как собственным', 
                   'any_vector_eigenvector', True, eigenvals, eigenvecs)

# 15. Пару отображений с AB ≠ BA
matrix_A = np.array([[1, 1], [0, 1]])
matrix_B = np.array([[1, 0], [1, 1]])
matrix_AB = np.dot(matrix_A, matrix_B)
matrix_BA = np.dot(matrix_B, matrix_A)

print(f"\n15. Матрица A:\n{matrix_A}")
print(f"Матрица B:\n{matrix_B}")
print(f"Матрица AB:\n{matrix_AB}")
print(f"Матрица BA:\n{matrix_BA}")
print(f"AB ≠ BA: {not np.array_equal(matrix_AB, matrix_BA)}")

# Визуализация A, B, AB, BA
apply_transformation(vertices, matrix_A, 'Матрица A', 'matrix_A')
apply_transformation(vertices, matrix_B, 'Матрица B', 'matrix_B')
apply_transformation(vertices, matrix_AB, 'Матрица AB', 'matrix_AB')
apply_transformation(vertices, matrix_BA, 'Матрица BA', 'matrix_BA')

# 16. Пару отображений с AB = BA
matrix_C = np.array([[2, 0], [0, 3]])
matrix_D = np.array([[1, 0], [0, 4]])
matrix_CD = np.dot(matrix_C, matrix_D)
matrix_DC = np.dot(matrix_D, matrix_C)

print(f"\n16. Матрица C:\n{matrix_C}")
print(f"Матрица D:\n{matrix_D}")
print(f"Матрица CD:\n{matrix_CD}")
print(f"Матрица DC:\n{matrix_DC}")
print(f"CD = DC: {np.array_equal(matrix_CD, matrix_DC)}")

# Визуализация C, D, CD, DC
apply_transformation(vertices, matrix_C, 'Матрица C', 'matrix_C')
apply_transformation(vertices, matrix_D, 'Матрица D', 'matrix_D')
apply_transformation(vertices, matrix_CD, 'Матрица CD', 'matrix_CD')
apply_transformation(vertices, matrix_DC, 'Матрица DC', 'matrix_DC')

print("\n=== ЗАДАНИЕ 2: Анализ ===")

# Анализ образов и ядер для пунктов 1, 2, 13, 14
def analyze_kernel_image(matrix, name):
    """Анализирует ядро и образ матрицы"""
    eigenvals, eigenvecs = get_eigenvalues_eigenvectors(matrix)
    det = np.linalg.det(matrix)
    rank = np.linalg.matrix_rank(matrix)
    
    print(f"\nАнализ {name}:")
    print(f"Определитель: {det}")
    print(f"Ранг: {rank}")
    print(f"Собственные значения: {eigenvals}")
    
    # Ядро (kernel) - пространство решений Ax = 0
    if det == 0:
        print("Ядро: нетривиальное (матрица вырождена)")
    else:
        print("Ядро: {0} (только нулевой вектор)")
    
    # Образ (image) - пространство значений Ax
    if rank == 2:
        print("Образ: вся плоскость R²")
    elif rank == 1:
        print("Образ: прямая")
    else:
        print("Образ: точка (начало координат)")

analyze_kernel_image(reflection_matrix, "отражения (пункт 1)")
analyze_kernel_image(projection_matrix, "проекции (пункт 2)")
analyze_kernel_image(matrix_13, "комплексных собственных значений (пункт 13)")
analyze_kernel_image(matrix_14, "скалярного отображения (пункт 14)")

# Анализ собственных значений для всех матриц
matrices_to_analyze = [
    (reflection_matrix, "отражения"),
    (projection_matrix, "проекции"),
    (rotation_matrix, "поворота"),
    (central_symmetry_matrix, "центральной симметрии"),
    (matrix_8, "обмена прямых"),
    (matrix_11, "с перпендикулярными собственными векторами"),
    (matrix_12, "без двух неколлинеарных собственных векторов"),
    (matrix_13, "с комплексными собственными значениями"),
    (matrix_14, "скалярного отображения"),
    (matrix_A, "A (некоммутативные)"),
    (matrix_B, "B (некоммутативные)"),
    (matrix_C, "C (коммутативные)"),
    (matrix_D, "D (коммутативные)")
]

print("\n=== Анализ собственных значений ===")
for matrix, name in matrices_to_analyze:
    eigenvals, eigenvecs = get_eigenvalues_eigenvectors(matrix)
    print(f"{name}: λ = {eigenvals}")

# Анализ определителей
matrices_for_det = [
    (reflection_matrix, "отражения"),
    (projection_matrix, "проекции"),
    (rotation_matrix, "поворота"),
    (central_symmetry_matrix, "центральной симметрии"),
    (combined_matrix, "отражение+поворот"),
    (matrix_9, "масштабирования"),
    (matrix_10, "эллиптического преобразования")
]

print("\n=== Анализ определителей ===")
for matrix, name in matrices_for_det:
    det = np.linalg.det(matrix)
    print(f"{name}: det = {det:.4f}")

# Анализ симметричности
print("\n=== Анализ симметричности ===")
symmetric_matrices = []
for matrix, name in matrices_to_analyze:
    is_symmetric = np.array_equal(matrix, matrix.T)
    print(f"{name}: {'симметрична' if is_symmetric else 'несимметрична'}")
    if is_symmetric:
        symmetric_matrices.append(name)

print(f"\nСимметричные матрицы: {symmetric_matrices}")

print("\nВсе задания выполнены!") 