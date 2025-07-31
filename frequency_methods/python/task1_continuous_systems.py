import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# Создаем папку для сохранения изображений
os.makedirs('../images/task1', exist_ok=True)

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Задаем два неколлинеарных вектора, не лежащих на координатных осях
v1 = np.array([1, 2])  # первый вектор
v2 = np.array([2, -1])  # второй вектор

print("Векторы v1 и v2:")
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"Скалярное произведение v1·v2 = {np.dot(v1, v2)}")
print()

# Функция для решения системы дифференциальных уравнений
def system_ode(x, t, A):
    """Система дифференциальных уравнений dx/dt = Ax"""
    return A @ x

# Функция для построения графиков
def plot_system(A, title, filename_base, eigenvalues, eigenvectors):
    """Построение графиков для одной системы"""
    
    # Время интегрирования
    t = np.linspace(0, 10, 1000)
    
    # Начальные условия (5 различных наборов)
    x0_list = [
        v1,                    # на собственном векторе v1
        v2,                    # на собственном векторе v2
        np.array([1, 0]),     # на оси x1
        np.array([0, 1]),     # на оси x2
        np.array([1, 1])      # произвольная точка
    ]
    
    # Цвета для разных траекторий
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    labels = ['v1', 'v2', '[1,0]', '[0,1]', '[1,1]']
    
    # График 1: x1(t) и x2(t)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for i, x0 in enumerate(x0_list):
        solution = odeint(system_ode, x0, t, args=(A,))
        plt.plot(t, solution[:, 0], color=colors[i], label=f'x1(t), {labels[i]}')
    plt.xlabel('Время t')
    plt.ylabel('x1(t)')
    plt.title(f'{title}\nx1(t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    for i, x0 in enumerate(x0_list):
        solution = odeint(system_ode, x0, t, args=(A,))
        plt.plot(t, solution[:, 1], color=colors[i], label=f'x2(t), {labels[i]}')
    plt.xlabel('Время t')
    plt.ylabel('x2(t)')
    plt.title(f'{title}\nx2(t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 2: Фазовая плоскость
    plt.subplot(1, 3, 3)
    for i, x0 in enumerate(x0_list):
        solution = odeint(system_ode, x0, t, args=(A,))
        plt.plot(solution[:, 0], solution[:, 1], color=colors[i], 
                label=f'{labels[i]}', linewidth=2)
        # Отмечаем начальную точку
        plt.plot(solution[0, 0], solution[0, 1], 'o', color=colors[i], markersize=8)
    
    # Показываем собственные векторы
    if eigenvectors is not None:
        for i, eigenvector in enumerate(eigenvectors):
            # Нормализуем для отображения
            norm_eigenvector = eigenvector / np.linalg.norm(eigenvector) * 2
            plt.arrow(0, 0, norm_eigenvector[0], norm_eigenvector[1], 
                     color='black', head_width=0.1, head_length=0.1, 
                     label=f'Собственный вектор {i+1}')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'{title}\nФазовая плоскость')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'../images/task1/{filename_base}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Выводим информацию о системе
    print(f"\n{title}")
    print(f"Матрица A:\n{A}")
    print(f"Собственные числа: {eigenvalues}")
    if eigenvectors is not None:
        print(f"Собственные векторы:\n{eigenvectors}")
    print()

# Система 1: Асимптотически устойчива, x(0) = v1 → x(t) ∈ Span{v1}, x(0) = v2 → x(t) ∈ Span{v2}
# Матрица с собственными числами λ1 = -1, λ2 = -2 и собственными векторами v1, v2
# A = P diag(-1, -2) P^(-1), где P = [v1, v2]
P = np.column_stack([v1, v2])
D = np.diag([-1, -2])
A1 = P @ D @ np.linalg.inv(P)

eigenvalues1, eigenvectors1 = np.linalg.eig(A1)
plot_system(A1, "Система 1: Асимптотически устойчива\nс инвариантными подпространствами", 
           "system1_asymptotically_stable", eigenvalues1, eigenvectors1)

# Система 2: Неустойчива, у матрицы A не существует двух неколлинеарных собственных векторов
# Матрица с собственным числом λ = 1 кратности 2 и одним собственным вектором
A2 = np.array([[1, 1],
               [0, 1]])  # Жорданова клетка

eigenvalues2, eigenvectors2 = np.linalg.eig(A2)
plot_system(A2, "Система 2: Неустойчива\nс дефектной матрицей", 
           "system2_unstable_defective", eigenvalues2, eigenvectors2)

# Система 3: Неустойчива, но x(0) = v1 → lim x(t) = 0
# Матрица с собственными числами λ1 = -1, λ2 = 1 и собственными векторами v1, v2
D3 = np.diag([-1, 1])
A3 = P @ D3 @ np.linalg.inv(P)

eigenvalues3, eigenvectors3 = np.linalg.eig(A3)
plot_system(A3, "Система 3: Неустойчива\nно v1 → 0", 
           "system3_unstable_v1_to_zero", eigenvalues3, eigenvectors3)

# Система 4: Асимптотически устойчива с комплексными собственными векторами
# Матрица с комплексными собственными числами λ = -0.5 ± 0.5i
# Собственные векторы: v1 ± v2i
A4 = np.array([[-0.5, -0.5],
               [0.5, -0.5]])

eigenvalues4, eigenvectors4 = np.linalg.eig(A4)
plot_system(A4, "Система 4: Асимптотически устойчива\nс комплексными собственными векторами", 
           "system4_asymptotically_stable_complex", eigenvalues4, eigenvectors4)

# Система 5: Неустойчива с теми же комплексными собственными векторами
# Матрица с комплексными собственными числами λ = 0.5 ± 0.5i
A5 = np.array([[0.5, -0.5],
               [0.5, 0.5]])

eigenvalues5, eigenvectors5 = np.linalg.eig(A5)
plot_system(A5, "Система 5: Неустойчива\nс комплексными собственными векторами", 
           "system5_unstable_complex", eigenvalues5, eigenvectors5)

# Система 6: Не асимптотически устойчива и не неустойчива (нейтрально устойчива)
# Матрица с чисто мнимыми собственными числами λ = ±i
A6 = np.array([[0, -1],
               [1, 0]])

eigenvalues6, eigenvectors6 = np.linalg.eig(A6)
plot_system(A6, "Система 6: Нейтрально устойчива\nс чисто мнимыми собственными числами", 
           "system6_neutrally_stable", eigenvalues6, eigenvectors6)

print("Моделирование непрерывных систем завершено!")
print("Все графики сохранены в папке images/task1/") 