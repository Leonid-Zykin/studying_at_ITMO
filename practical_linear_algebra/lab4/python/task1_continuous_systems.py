import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import os

# Создаем папку для изображений если её нет
os.makedirs('../images/task1', exist_ok=True)

# Задаем два неколлинеарных вектора, не лежащих на координатных осях
v1 = np.array([1, 1])  # вектор (1, 1)
v2 = np.array([-1, 2])  # вектор (-1, 2)

print(f"Вектор v1 = {v1}")
print(f"Вектор v2 = {v2}")
print(f"Проверка коллинеарности: {np.cross(v1, v2)}")  # должно быть не равно 0

def system_1(x, t, A):
    """Система 1: Асимптотически устойчива, собственные векторы v1 и v2"""
    return A @ x

def system_2(x, t, A):
    """Система 2: Неустойчива, нет двух неколлинеарных собственных векторов"""
    return A @ x

def system_3(x, t, A):
    """Система 3: Неустойчива, но x(0) = v1 -> lim x(t) = 0"""
    return A @ x

def system_4(x, t, A):
    """Система 4: Асимптотически устойчива, комплексные собственные векторы"""
    return A @ x

def system_5(x, t, A):
    """Система 5: Неустойчива, те же собственные векторы что в системе 4"""
    return A @ x

def system_6(x, t, A):
    """Система 6: Не асимптотически устойчива и не неустойчива, те же собственные векторы"""
    return A @ x

# Создаем матрицы для каждой системы
# Система 1: Асимптотически устойчива, собственные векторы v1 и v2
# Нужно найти матрицу A такую, что Av1 = λ1*v1 и Av2 = λ2*v2, где λ1, λ2 < 0
# Пусть λ1 = -1, λ2 = -2
lambda1, lambda2 = -1, -2
# Решаем систему уравнений для нахождения элементов матрицы A
# A[0,0]*v1[0] + A[0,1]*v1[1] = λ1*v1[0]
# A[1,0]*v1[0] + A[1,1]*v1[1] = λ1*v1[1]
# A[0,0]*v2[0] + A[0,1]*v2[1] = λ2*v2[0]
# A[1,0]*v2[0] + A[1,1]*v2[1] = λ2*v2[1]

# Решаем систему линейных уравнений
V = np.array([[v1[0], v1[1], 0, 0],
              [0, 0, v1[0], v1[1]],
              [v2[0], v2[1], 0, 0],
              [0, 0, v2[0], v2[1]]])

b = np.array([lambda1*v1[0], lambda1*v1[1], lambda2*v2[0], lambda2*v2[1]])

A1_elements = np.linalg.solve(V, b)
A1 = np.array([[A1_elements[0], A1_elements[1]],
                [A1_elements[2], A1_elements[3]]])

print(f"Матрица A1 для системы 1:\n{A1}")
print(f"Собственные значения A1: {np.linalg.eigvals(A1)}")
print(f"Собственные векторы A1:\n{np.linalg.eig(A1)[1]}")

# Система 2: Неустойчива, нет двух неколлинеарных собственных векторов
# Используем жорданову клетку
A2 = np.array([[-1, 1],
                [0, -1]])

print(f"\nМатрица A2 для системы 2:\n{A2}")
print(f"Собственные значения A2: {np.linalg.eigvals(A2)}")
print(f"Собственные векторы A2:\n{np.linalg.eig(A2)[1]}")

# Система 3: Неустойчива, но x(0) = v1 -> lim x(t) = 0
# Используем матрицу с одним положительным и одним отрицательным собственным значением
# v1 должен быть собственным вектором для отрицательного собственного значения
A3 = np.array([[1, 0],
                [0, -2]])

print(f"\nМатрица A3 для системы 3:\n{A3}")
print(f"Собственные значения A3: {np.linalg.eigvals(A3)}")
print(f"Собственные векторы A3:\n{np.linalg.eig(A3)[1]}")

# Система 4: Асимптотически устойчива, комплексные собственные векторы
# Используем матрицу с комплексными собственными значениями с отрицательной вещественной частью
A4 = np.array([[-1, -1],
                [1, -1]])

print(f"\nМатрица A4 для системы 4:\n{A4}")
print(f"Собственные значения A4: {np.linalg.eigvals(A4)}")
print(f"Собственные векторы A4:\n{np.linalg.eig(A4)[1]}")

# Система 5: Неустойчива, те же собственные векторы что в системе 4
# Используем ту же структуру, но с положительной вещественной частью
A5 = np.array([[1, -1],
                [1, 1]])

print(f"\nМатрица A5 для системы 5:\n{A5}")
print(f"Собственные значения A5: {np.linalg.eigvals(A5)}")
print(f"Собственные векторы A5:\n{np.linalg.eig(A5)[1]}")

# Система 6: Не асимптотически устойчива и не неустойчива, те же собственные векторы
# Используем мнимые собственные значения
A6 = np.array([[0, -1],
                [1, 0]])

print(f"\nМатрица A6 для системы 6:\n{A6}")
print(f"Собственные значения A6: {np.linalg.eigvals(A6)}")
print(f"Собственные векторы A6:\n{np.linalg.eig(A6)[1]}")

# Функция для решения системы и построения графиков
def solve_and_plot(system_func, A, system_name, t_span=np.linspace(0, 10, 1000)):
    """Решает систему и строит графики"""
    
    # Начальные условия
    x0_1 = v1  # Начальное условие на собственном векторе
    x0_2 = v2  # Начальное условие на собственном векторе
    x0_3 = np.array([2, 0])  # Произвольное начальное условие
    x0_4 = np.array([0, 2])  # Произвольное начальное условие
    x0_5 = np.array([1, -1])  # Произвольное начальное условие
    
    initial_conditions = [x0_1, x0_2, x0_3, x0_4, x0_5]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    labels = ['v1', 'v2', 'x0_3', 'x0_4', 'x0_5']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Решаем систему для каждого начального условия
    for i, (x0, color, label) in enumerate(zip(initial_conditions, colors, labels)):
        solution = odeint(system_func, x0, t_span, args=(A,))
        
        # График x1(t)
        ax1.plot(t_span, solution[:, 0], color=color, label=f'{label}: x1(t)')
        
        # График x2(t)
        ax2.plot(t_span, solution[:, 1], color=color, label=f'{label}: x2(t)')
        
        # Фазовый портрет
        ax3.plot(solution[:, 0], solution[:, 1], color=color, label=label)
        
        # Начальная точка
        ax3.plot(solution[0, 0], solution[0, 1], 'o', color=color, markersize=8)
    
    # Настройка графиков
    ax1.set_xlabel('t')
    ax1.set_ylabel('x1(t)')
    ax1.set_title(f'{system_name} - График x1(t)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('t')
    ax2.set_ylabel('x2(t)')
    ax2.set_title(f'{system_name} - График x2(t)')
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_title(f'{system_name} - Фазовый портрет')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # Добавляем собственные векторы
    eigenvals, eigenvecs = np.linalg.eig(A)
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvals, eigenvecs.T)):
        # Нормализуем собственный вектор и берем только вещественную часть для отображения
        eigenvec_real = eigenvec.real / np.linalg.norm(eigenvec.real)
        # Рисуем собственный вектор
        ax3.arrow(0, 0, eigenvec_real[0], eigenvec_real[1], 
                  head_width=0.1, head_length=0.1, fc='black', ec='black', 
                  label=f'Собственный вектор λ={eigenval:.2f}')
    
    # График собственных значений на комплексной плоскости
    for i, eigenval in enumerate(eigenvals):
        ax4.plot(eigenval.real, eigenval.imag, 'o', markersize=10, 
                label=f'λ{i+1} = {eigenval:.2f}')
    
    ax4.set_xlabel('Re(λ)')
    ax4.set_ylabel('Im(λ)')
    ax4.set_title(f'{system_name} - Собственные значения')
    ax4.legend()
    ax4.grid(True)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../images/task1/{system_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Решаем и строим графики для всех систем
systems = [
    (system_1, A1, "Система 1 - Асимптотически устойчива"),
    (system_2, A2, "Система 2 - Неустойчива, нет неколлинеарных собственных векторов"),
    (system_3, A3, "Система 3 - Неустойчива, но v1 -> 0"),
    (system_4, A4, "Система 4 - Асимптотически устойчива, комплексные собственные векторы"),
    (system_5, A5, "Система 5 - Неустойчива, комплексные собственные векторы"),
    (system_6, A6, "Система 6 - Не асимптотически устойчива и не неустойчива")
]

for system_func, A, name in systems:
    print(f"\nРешаем {name}...")
    solve_and_plot(system_func, A, name)

print("\nВсе графики сохранены в папке ../images/task1/") 