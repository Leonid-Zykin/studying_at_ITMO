import numpy as np
import matplotlib.pyplot as plt
import os

# Создаем папку для сохранения изображений
os.makedirs('../images/task2', exist_ok=True)

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Функция для построения графиков дискретной системы
def plot_discrete_system(A, title, filename_base, eigenvalues):
    """Построение графиков для дискретной системы"""
    
    # Начальные условия
    x0 = np.array([1, 1])
    
    # Количество итераций
    k_max = 50
    k = np.arange(k_max + 1)
    
    # Вычисляем траекторию
    x = np.zeros((k_max + 1, 2))
    x[0] = x0
    
    for i in range(k_max):
        x[i + 1] = A @ x[i]
    
    # График 1: x1(k) и x2(k)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(k, x[:, 0], 'bo-', markersize=4, label='x1(k)')
    plt.xlabel('k')
    plt.ylabel('x1(k)')
    plt.title(f'{title}\nx1(k)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(k, x[:, 1], 'ro-', markersize=4, label='x2(k)')
    plt.xlabel('k')
    plt.ylabel('x2(k)')
    plt.title(f'{title}\nx2(k)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 2: Фазовая плоскость
    plt.subplot(1, 3, 3)
    plt.plot(x[:, 0], x[:, 1], 'go-', markersize=4, label='Траектория')
    plt.plot(x[0, 0], x[0, 1], 'ko', markersize=8, label='Начальная точка')
    
    # Показываем собственные векторы
    eigenvalues_A, eigenvectors_A = np.linalg.eig(A)
    for i, eigenvector in enumerate(eigenvectors_A):
        real_eigenvector = np.real(eigenvector)
        if np.linalg.norm(real_eigenvector) > 1e-10:
            norm_eigenvector = real_eigenvector / np.linalg.norm(real_eigenvector) * 2
            plt.arrow(0, 0, norm_eigenvector[0], norm_eigenvector[1], 
                     color='black', head_width=0.1, head_length=0.1, 
                     label=f'Собственный вектор {i+1} (Re)')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'{title}\nФазовая плоскость')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'../images/task2/{filename_base}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Выводим информацию о системе
    print(f"\n{title}")
    print(f"Матрица A:\n{A}")
    print(f"Собственные числа: {eigenvalues}")
    print(f"Модули собственных чисел: {np.abs(eigenvalues)}")
    print()

# Функция для отображения собственных чисел на комплексной плоскости
def plot_eigenvalues_complex_plane(eigenvalues_list, titles, filename):
    """Отображение собственных чисел на комплексной плоскости"""
    
    plt.figure(figsize=(12, 8))
    
    # Единичная окружность
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='|λ| = 1')
    
    # Оси координат
    plt.axhline(y=0, color='k', alpha=0.3)
    plt.axvline(x=0, color='k', alpha=0.3)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 
              'olive', 'cyan', 'magenta', 'yellow']
    
    for i, eigenvalues in enumerate(eigenvalues_list):
        color = colors[i % len(colors)]
        for j, eigenvalue in enumerate(eigenvalues):
            plt.plot(np.real(eigenvalue), np.imag(eigenvalue), 'o', 
                    color=color, markersize=8, label=f'{titles[i]}' if j == 0 else "")
    
    plt.xlabel('Re(λ)')
    plt.ylabel('Im(λ)')
    plt.title('Собственные числа на комплексной плоскости')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'../images/task2/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Система 1: λ1,2 = -1
A1 = np.array([[-1, 0],
               [0, -1]])

eigenvalues1 = np.array([-1, -1])
plot_discrete_system(A1, "Система 1: λ1,2 = -1", "system1_lambda_minus1", eigenvalues1)

# Система 2: λ1,2 = -1/√2 ± (1/√2)i
A2 = np.array([[-1/np.sqrt(2), -1/np.sqrt(2)],
               [1/np.sqrt(2), -1/np.sqrt(2)]])

eigenvalues2 = np.array([-1/np.sqrt(2) + 1j/np.sqrt(2), -1/np.sqrt(2) - 1j/np.sqrt(2)])
plot_discrete_system(A2, "Система 2: λ1,2 = -1/√2 ± (1/√2)i", "system2_lambda_complex_stable", eigenvalues2)

# Система 3: λ1,2 = ±i
A3 = np.array([[0, -1],
               [1, 0]])

eigenvalues3 = np.array([1j, -1j])
plot_discrete_system(A3, "Система 3: λ1,2 = ±i", "system3_lambda_pure_imaginary", eigenvalues3)

# Система 4: λ1,2 = 1/√2 ± (1/√2)i
A4 = np.array([[1/np.sqrt(2), -1/np.sqrt(2)],
               [1/np.sqrt(2), 1/np.sqrt(2)]])

eigenvalues4 = np.array([1/np.sqrt(2) + 1j/np.sqrt(2), 1/np.sqrt(2) - 1j/np.sqrt(2)])
plot_discrete_system(A4, "Система 4: λ1,2 = 1/√2 ± (1/√2)i", "system4_lambda_complex_unstable", eigenvalues4)

# Система 5: λ1,2 = 1
A5 = np.array([[1, 0],
               [0, 1]])

eigenvalues5 = np.array([1, 1])
plot_discrete_system(A5, "Система 5: λ1,2 = 1", "system5_lambda_plus1", eigenvalues5)

# Системы 6-8: Те же собственные числа, умноженные на c = 0.5 (0 < c < 1)
c = 0.5

A6 = c * A1  # λ1,2 = -0.5
eigenvalues6 = c * eigenvalues1
plot_discrete_system(A6, f"Система 6: λ1,2 = {c}·(-1) = -{c}", "system6_lambda_minus1_scaled", eigenvalues6)

A8 = c * A3  # λ1,2 = ±0.5i
eigenvalues8 = c * eigenvalues3
plot_discrete_system(A8, f"Система 8: λ1,2 = {c}·(±i) = ±{c}i", "system8_lambda_pure_imaginary_scaled", eigenvalues8)

A10 = c * A5  # λ1,2 = 0.5
eigenvalues10 = c * eigenvalues5
plot_discrete_system(A10, f"Система 10: λ1,2 = {c}·1 = {c}", "system10_lambda_plus1_scaled", eigenvalues10)

# Системы 9-11: Те же собственные числа, умноженные на d = 2 (d > 1)
d = 2.0

A9 = d * A1  # λ1,2 = -2
eigenvalues9 = d * eigenvalues1
plot_discrete_system(A9, f"Система 9: λ1,2 = {d}·(-1) = -{d}", "system9_lambda_minus1_scaled_up", eigenvalues9)

A11 = d * A3  # λ1,2 = ±2i
eigenvalues11 = d * eigenvalues3
plot_discrete_system(A11, f"Система 11: λ1,2 = {d}·(±i) = ±{d}i", "system11_lambda_pure_imaginary_scaled_up", eigenvalues11)

A13 = d * A5  # λ1,2 = 2
eigenvalues13 = d * eigenvalues5
plot_discrete_system(A13, f"Система 13: λ1,2 = {d}·1 = {d}", "system13_lambda_plus1_scaled_up", eigenvalues13)

# Система 12: λ1,2 = 0
A12 = np.array([[0, 0],
                [0, 0]])

eigenvalues12 = np.array([0, 0])
plot_discrete_system(A12, "Система 12: λ1,2 = 0", "system12_lambda_zero", eigenvalues12)

# Отображение всех собственных чисел на комплексной плоскости
eigenvalues_list = [
    eigenvalues1, eigenvalues2, eigenvalues3, eigenvalues4, eigenvalues5,
    eigenvalues6, eigenvalues8, eigenvalues10, eigenvalues9, eigenvalues11, eigenvalues13, eigenvalues12
]

titles = [
    "λ1,2 = -1", "λ1,2 = -1/√2 ± (1/√2)i", "λ1,2 = ±i", 
    "λ1,2 = 1/√2 ± (1/√2)i", "λ1,2 = 1",
    f"λ1,2 = {c}·(-1)", f"λ1,2 = {c}·(±i)", f"λ1,2 = {c}·1",
    f"λ1,2 = {d}·(-1)", f"λ1,2 = {d}·(±i)", f"λ1,2 = {d}·1", "λ1,2 = 0"
]

plot_eigenvalues_complex_plane(eigenvalues_list, titles, "eigenvalues_complex_plane")

print("Моделирование дискретных систем завершено!")
print("Все графики сохранены в папке images/task2/") 