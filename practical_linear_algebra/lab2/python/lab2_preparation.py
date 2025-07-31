import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches

# Выбор чисел a, b, c, d (все различны и не равны 0, ±1)
a = 3
b = 5
c = 7
d = 11

print(f"Выбранные числа: a={a}, b={b}, c={c}, d={d}")

# Создание многоугольника (неправильный шестиугольник)
vertices = np.array([
    [0, 0],    # вершина 1
    [2, 1],    # вершина 2
    [3, 3],    # вершина 3
    [2, 4],    # вершина 4
    [0, 3],    # вершина 5
    [-1, 1]    # вершина 6
])

# Создание графика
fig, ax = plt.subplots(figsize=(10, 8))

# Отрисовка многоугольника
polygon = Polygon(vertices, facecolor='lightblue', edgecolor='blue', linewidth=2)
ax.add_patch(polygon)

# Отрисовка вершин
ax.scatter(vertices[:, 0], vertices[:, 1], color='red', s=100, zorder=5)

# Подписи вершин
for i, vertex in enumerate(vertices):
    ax.annotate(f'V{i+1}', (vertex[0], vertex[1]), 
                xytext=(5, 5), textcoords='offset points', fontsize=12)

# Настройка осей
ax.set_xlim(-2, 4)
ax.set_ylim(-1, 5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Исходный многоугольник')

# Добавление координатных осей
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig('images/original_polygon.png', dpi=300, bbox_inches='tight')
plt.show()

print("Многоугольник создан и сохранен в lab2/images/original_polygon.png")
print(f"Вершины многоугольника: {vertices}") 