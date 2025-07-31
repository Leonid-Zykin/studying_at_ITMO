import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# Создаем папку для сохранения изображений
os.makedirs('../images/task1', exist_ok=True)

# Отключаем интерактивный режим matplotlib
plt.ioff()

def create_cube_vertices():
    """Создание вершин кубика в однородных координатах"""
    # Вершины кубика в однородных координатах (x, y, z, w)
    vertices = np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],    # x координаты
        [-1, -1, 1, 1, -1, -1, 1, 1],    # y координаты
        [-1, -1, -1, -1, 1, 1, 1, 1],    # z координаты
        [1, 1, 1, 1, 1, 1, 1, 1]         # w координаты (однородные)
    ])
    return vertices

def create_cube_faces():
    """Создание граней кубика"""
    # Грани кубика (индексы вершин для каждой грани)
    faces = np.array([
        [0, 1, 5, 4],  # Передняя грань
        [1, 2, 6, 5],  # Правая грань
        [2, 3, 7, 6],  # Задняя грань
        [3, 0, 4, 7],  # Левая грань
        [0, 1, 2, 3],  # Нижняя грань
        [4, 5, 6, 7]   # Верхняя грань
    ])
    return faces

def draw_cube(vertices, faces, color='blue', title='Кубик', filename='cube'):
    """Отрисовка кубика"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    # Преобразуем однородные координаты в декартовы
    # (x, y, z, w) -> (x/w, y/w, z/w)
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
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    
    # Устанавливаем углы обзора
    ax.view_init(azim=-37.5, elev=30)
    
    # Добавляем сетку
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../images/task1/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_homogeneous_coordinates():
    """Анализ однородных координат"""
    print("Анализ однородных координат:")
    print("=" * 50)
    
    vertices = create_cube_vertices()
    
    print("Вершины кубика в однородных координатах:")
    print("Формат: (x, y, z, w)")
    for i in range(vertices.shape[1]):
        x, y, z, w = vertices[:, i]
        print(f"Вершина {i}: ({x}, {y}, {z}, {w})")
    
    print("\nПреобразование в декартовы координаты:")
    print("(x, y, z, w) -> (x/w, y/w, z/w)")
    cartesian = (vertices[:3, :] / vertices[3, :]).T
    for i in range(cartesian.shape[0]):
        x, y, z = cartesian[i]
        print(f"Вершина {i}: ({x:.1f}, {y:.1f}, {z:.1f})")
    
    print("\nПочему используются однородные координаты?")
    print("1. Позволяют представлять аффинные преобразования как матричные")
    print("2. Упрощают композицию преобразований")
    print("3. Позволяют представлять точки на бесконечности")
    print("4. Необходимы для перспективных преобразований")

def create_other_shapes():
    """Создание других фигур"""
    print("\nСоздание других фигур:")
    print("=" * 50)
    
    # Тетраэдр
    tetrahedron_vertices = np.array([
        [0, 1, -1, 0],      # x
        [0, 0, 0, 1],       # y
        [1, -1, -1, -1],    # z
        [1, 1, 1, 1]        # w
    ])
    
    tetrahedron_faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])
    
    print("Тетраэдр создан с 4 вершинами и 4 треугольными гранями")
    
    # Пирамида с квадратным основанием
    pyramid_vertices = np.array([
        [-1, 1, 0, -1, 0],   # x
        [-1, -1, 0, 1, 0],   # y
        [-1, -1, 1, -1, 2],  # z
        [1, 1, 1, 1, 1]      # w
    ])
    
    pyramid_faces = np.array([
        [0, 1, 2, 3],  # Основание
        [0, 1, 4],      # Боковая грань 1
        [1, 2, 4],      # Боковая грань 2
        [2, 3, 4],      # Боковая грань 3
        [3, 0, 4]       # Боковая грань 4
    ], dtype=object)
    
    print("Пирамида создана с 5 вершинами и 5 гранями")
    
    return tetrahedron_vertices, tetrahedron_faces, pyramid_vertices, pyramid_faces

def main():
    print("Задание 1: Создание кубика")
    print("=" * 50)
    
    # Создаем кубик
    vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    print(f"Создан кубик с {vertices.shape[1]} вершинами и {faces.shape[0]} гранями")
    
    # Отрисовываем кубик
    draw_cube(vertices, faces, 'blue', 'Исходный кубик', 'original_cube')
    print("Кубик отрисован и сохранен в images/task1/original_cube.png")
    
    # Анализируем однородные координаты
    analyze_homogeneous_coordinates()
    
    # Создаем другие фигуры
    tetra_vertices, tetra_faces, pyramid_vertices, pyramid_faces = create_other_shapes()
    
    # Отрисовываем тетраэдр
    draw_cube(tetra_vertices, tetra_faces, 'red', 'Тетраэдр', 'tetrahedron')
    print("Тетраэдр отрисован и сохранен в images/task1/tetrahedron.png")
    
    # Отрисовываем пирамиду
    draw_cube(pyramid_vertices, pyramid_faces, 'green', 'Пирамида', 'pyramid')
    print("Пирамида отрисована и сохранена в images/task1/pyramid.png")
    
    print("\nЗадание 1 завершено!")
    print("Все изображения сохранены в папке images/task1/")

if __name__ == "__main__":
    main() 