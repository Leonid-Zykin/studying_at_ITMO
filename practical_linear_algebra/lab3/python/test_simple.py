import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

print("Начинаем создание простого кубика...")

# Создание вершин кубика
vertices_cube = np.array([
    [-1, 1, 1, -1, -1, 1, 1, -1],
    [-1, -1, 1, 1, -1, -1, 1, 1],
    [-1, -1, -1, -1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
])

# Определение граней
faces_cube = np.array([
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [0, 1, 2, 3],
    [4, 5, 6, 7]
])

def draw_shape(vertices, faces, color, alpha=0.7):
    vertices_cartesian = (vertices[:3, :] / vertices[3, :]).T
    poly3d = Poly3DCollection(vertices_cartesian[faces], 
                             facecolors=color, 
                             edgecolors='black', 
                             linewidths=0.5,
                             alpha=alpha)
    return poly3d

print("Создаем фигуру...")
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

print("Отрисовываем кубик...")
poly3d = draw_shape(vertices_cube, faces_cube, 'blue')
ax.add_collection3d(poly3d)

ax.set_box_aspect([1, 1, 1])
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Тестовый кубик')

print("Сохраняем изображение...")
plt.savefig('images/task1/test_cube.png', dpi=300, bbox_inches='tight')
plt.close()

print("Готово! Изображение сохранено в images/task1/test_cube.png") 