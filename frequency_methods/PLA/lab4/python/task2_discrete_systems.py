import numpy as np
import matplotlib.pyplot as plt
import os

# Создаем папку для сохранения изображений
os.makedirs('../images/task2', exist_ok=True)

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Вспомогательные функции

def similar_matrix(J: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Возвращает матрицу A = P J P^{-1}."""
    return P @ J @ np.linalg.inv(P)

# Фиксируем невырожденную матрицу подобия (не диагональную и с det!=0)
P = np.array([[1.0, 2.0],
              [3.0, 5.0]])  # det = -1

# Функция для построения графиков дискретной системы
def plot_discrete_system(A, title, filename_base, eigenvalues):
    """Построение графиков для дискретной системы"""
    
    # Начальные условия
    x0 = np.array([1.0, 1.0])
    
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
    for i in range(eigenvectors_A.shape[1]):
        eigenvector = eigenvectors_A[:, i]
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
    print(f"Собственные числа (заданные): {eigenvalues}")
    print(f"Собственные числа (фактические): {np.linalg.eigvals(A)}")
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

# Конструкции через подобие, чтобы A не была в диагональном или жордановом виде

# 1) λ1,2 = -1 (одинарный жордановский блок в каноне) → берём J как Ж-клетку и применяем подобие
J1 = np.array([[-1.0, 1.0],
               [ 0.0,-1.0]])
A1 = similar_matrix(J1, P)
eigenvalues1 = np.array([-1+0j, -1+0j])
plot_discrete_system(A1, "Система 1: λ1,2 = -1", "system1_lambda_minus1", eigenvalues1)

# 2) λ1,2 = -1/√2 ± (1/√2)i (модуль=1)
a2 = -1/np.sqrt(2)
b2 =  1/np.sqrt(2)
J2 = np.array([[a2, -b2],
               [b2,  a2]])
A2 = similar_matrix(J2, P)
eigenvalues2 = np.array([a2 + 1j*b2, a2 - 1j*b2])
plot_discrete_system(A2, "Система 2: λ1,2 = -1/√2 ± (1/√2)i", "system2_lambda_complex_stable", eigenvalues2)

# 3) λ1,2 = ±i
a3, b3 = 0.0, 1.0
J3 = np.array([[a3, -b3],
               [b3,  a3]])
A3 = similar_matrix(J3, P)
eigenvalues3 = np.array([1j, -1j])
plot_discrete_system(A3, "Система 3: λ1,2 = ±i", "system3_lambda_pure_imaginary", eigenvalues3)

# 4) λ1,2 = 1/√2 ± (1/√2)i (модуль=1)
a4 = 1/np.sqrt(2)
b4 = 1/np.sqrt(2)
J4 = np.array([[a4, -b4],
               [b4,  a4]])
A4 = similar_matrix(J4, P)
eigenvalues4 = np.array([a4 + 1j*b4, a4 - 1j*b4])
plot_discrete_system(A4, "Система 4: λ1,2 = 1/√2 ± (1/√2)i", "system4_lambda_complex_unstable", eigenvalues4)

# 5) λ1,2 = 1 (аналогично п.1)
J5 = np.array([[1.0, 1.0],
               [0.0, 1.0]])
A5 = similar_matrix(J5, P)
eigenvalues5 = np.array([1+0j, 1+0j])
plot_discrete_system(A5, "Система 5: λ1,2 = 1", "system5_lambda_plus1", eigenvalues5)

# Масштабирования c=0.5 (пп.6-8) и d=2.0 (пп.9-11)
c = 0.5
d = 2.0

# 6) c·(-1)
A6 = similar_matrix(c * J1, P)
eigenvalues6 = c * eigenvalues1
plot_discrete_system(A6, f"Система 6: λ1,2 = {c}·(-1)", "system6_lambda_minus1_scaled", eigenvalues6)

# 7) c·(±i)
A7 = similar_matrix(c * J3, P)
eigenvalues7 = c * eigenvalues3
plot_discrete_system(A7, f"Система 7: λ1,2 = {c}·(±i)", "system7_lambda_pure_imaginary_scaled", eigenvalues7)

# 8) c·(1)
A8 = similar_matrix(c * J5, P)
eigenvalues8 = c * eigenvalues5
plot_discrete_system(A8, f"Система 8: λ1,2 = {c}·1", "system8_lambda_plus1_scaled", eigenvalues8)

# 9) d·(-1)
A9 = similar_matrix(d * J1, P)
eigenvalues9 = d * eigenvalues1
plot_discrete_system(A9, f"Система 9: λ1,2 = {d}·(-1)", "system9_lambda_minus1_scaled_up", eigenvalues9)

# 10) d·(±i)
A10 = similar_matrix(d * J3, P)
eigenvalues10 = d * eigenvalues3
plot_discrete_system(A10, f"Система 10: λ1,2 = {d}·(±i)", "system10_lambda_pure_imaginary_scaled_up", eigenvalues10)

# 11) d·1
A11 = similar_matrix(d * J5, P)
eigenvalues11 = d * eigenvalues5
plot_discrete_system(A11, f"Система 11: λ1,2 = {d}·1", "system11_lambda_plus1_scaled_up", eigenvalues11)

# 12) λ1,2 = 0 (жордановский блок для нулевого собственного числа)
J12 = np.array([[0.0, 1.0],
                [0.0, 0.0]])
A12 = similar_matrix(J12, P)
eigenvalues12 = np.array([0+0j, 0+0j])
plot_discrete_system(A12, "Система 12: λ1,2 = 0", "system12_lambda_zero", eigenvalues12)

# Отображение всех собственных чисел на комплексной плоскости
eigenvalues_list = [
    eigenvalues1, eigenvalues2, eigenvalues3, eigenvalues4, eigenvalues5,
    eigenvalues6, eigenvalues7, eigenvalues8, eigenvalues9, eigenvalues10, eigenvalues11, eigenvalues12
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