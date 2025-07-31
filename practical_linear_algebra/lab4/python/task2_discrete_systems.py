import numpy as np
import matplotlib.pyplot as plt
import os

# Создаем папку для изображений если её нет
os.makedirs('../images/task2', exist_ok=True)

def create_matrix_with_eigenvalues(eigenvals):
    """Создает матрицу с заданными собственными значениями"""
    # Используем формулу A = P * D * P^(-1), где D - диагональная матрица с собственными значениями
    # P - матрица перехода (не диагональная и не жорданова)
    
    # Создаем случайную недиагональную матрицу P
    np.random.seed(42)  # для воспроизводимости
    P = np.random.rand(2, 2)
    # Делаем P невырожденной
    P[0, 1] = P[0, 1] + 0.5
    P[1, 0] = P[1, 0] + 0.3
    
    # Создаем диагональную матрицу D с собственными значениями
    D = np.diag(eigenvals)
    
    # Вычисляем A = P * D * P^(-1)
    A = P @ D @ np.linalg.inv(P)
    
    return A

def discrete_system(x, A):
    """Дискретная динамическая система x(k+1) = A*x(k)"""
    return A @ x

def simulate_discrete_system(A, x0, k_max=50):
    """Симулирует дискретную систему"""
    x = np.zeros((k_max + 1, 2))
    x[0] = x0
    
    for k in range(k_max):
        x[k + 1] = discrete_system(x[k], A).real  # Берем только вещественную часть
    
    return x

def plot_discrete_system(A, system_name, x0=np.array([1, 1]), k_max=50):
    """Строит графики для дискретной системы"""
    
    # Симулируем систему
    x = simulate_discrete_system(A, x0, k_max)
    k_values = np.arange(k_max + 1)
    
    # Собственные значения
    eigenvals = np.linalg.eigvals(A)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # График x1(k)
    ax1.plot(k_values, x[:, 0], 'b-o', markersize=4, label='x1(k)')
    ax1.set_xlabel('k')
    ax1.set_ylabel('x1(k)')
    ax1.set_title(f'{system_name} - График x1(k)')
    ax1.legend()
    ax1.grid(True)
    
    # График x2(k)
    ax2.plot(k_values, x[:, 1], 'r-o', markersize=4, label='x2(k)')
    ax2.set_xlabel('k')
    ax2.set_ylabel('x2(k)')
    ax2.set_title(f'{system_name} - График x2(k)')
    ax2.legend()
    ax2.grid(True)
    
    # Фазовый портрет
    ax3.plot(x[:, 0], x[:, 1], 'g-o', markersize=4, label='Траектория')
    ax3.plot(x[0, 0], x[0, 1], 'ko', markersize=8, label='Начальная точка')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_title(f'{system_name} - Фазовый портрет')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # Собственные значения на комплексной плоскости
    for i, eigenval in enumerate(eigenvals):
        ax4.plot(eigenval.real, eigenval.imag, 'o', markersize=10, 
                label=f'λ{i+1} = {eigenval:.3f}')
    
    # Добавляем единичную окружность
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.5)
    ax4.add_patch(circle)
    
    ax4.set_xlabel('Re(λ)')
    ax4.set_ylabel('Im(λ)')
    ax4.set_title(f'{system_name} - Собственные значения')
    ax4.legend()
    ax4.grid(True)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    # Создаем простое имя файла
    if "система 1" in system_name.lower():
        filename = "system_1_lambda_minus_1.png"
    elif "система 2" in system_name.lower():
        filename = "system_2_lambda_minus_1_sqrt2_pm_i_sqrt2.png"
    elif "система 3" in system_name.lower():
        filename = "system_3_lambda_pm_i.png"
    elif "система 4" in system_name.lower():
        filename = "system_4_lambda_1_sqrt2_pm_i_sqrt2.png"
    elif "система 5" in system_name.lower():
        filename = "system_5_lambda_1.png"
    elif "система 6" in system_name.lower():
        filename = "system_6_lambda_minus_0_5.png"
    elif "система 7" in system_name.lower():
        filename = "system_7_lambda_pm_0_5i.png"
    elif "система 8" in system_name.lower():
        filename = "system_8_lambda_0_5.png"
    elif "система 9" in system_name.lower():
        filename = "system_9_lambda_minus_1_5.png"
    elif "система 10" in system_name.lower():
        filename = "system_10_lambda_pm_1_5i.png"
    elif "система 11" in system_name.lower():
        filename = "system_11_lambda_1_5.png"
    elif "система 12" in system_name.lower():
        filename = "system_12_lambda_0.png"
    else:
        filename = "system_unknown.png"
    
    plt.savefig(f'../images/task2/{filename}', 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Создаем системы с заданными собственными значениями

# 1. λ1,2 = -1
eigenvals_1 = [-1, -1]
A1 = create_matrix_with_eigenvalues(eigenvals_1)
print(f"Система 1 - λ1,2 = -1")
print(f"Матрица A1:\n{A1}")
print(f"Собственные значения A1: {np.linalg.eigvals(A1)}")
print()

# 2. λ1,2 = -1/√2 ± 1/√2*i
eigenvals_2 = [-1/np.sqrt(2) + 1j/np.sqrt(2), -1/np.sqrt(2) - 1j/np.sqrt(2)]
A2 = create_matrix_with_eigenvalues(eigenvals_2)
print(f"Система 2 - λ1,2 = -1/√2 ± 1/√2*i")
print(f"Матрица A2:\n{A2}")
print(f"Собственные значения A2: {np.linalg.eigvals(A2)}")
print()

# 3. λ1,2 = ±i
eigenvals_3 = [1j, -1j]
A3 = create_matrix_with_eigenvalues(eigenvals_3)
print(f"Система 3 - λ1,2 = ±i")
print(f"Матрица A3:\n{A3}")
print(f"Собственные значения A3: {np.linalg.eigvals(A3)}")
print()

# 4. λ1,2 = 1/√2 ± 1/√2*i
eigenvals_4 = [1/np.sqrt(2) + 1j/np.sqrt(2), 1/np.sqrt(2) - 1j/np.sqrt(2)]
A4 = create_matrix_with_eigenvalues(eigenvals_4)
print(f"Система 4 - λ1,2 = 1/√2 ± 1/√2*i")
print(f"Матрица A4:\n{A4}")
print(f"Собственные значения A4: {np.linalg.eigvals(A4)}")
print()

# 5. λ1,2 = 1
eigenvals_5 = [1, 1]
A5 = create_matrix_with_eigenvalues(eigenvals_5)
print(f"Система 5 - λ1,2 = 1")
print(f"Матрица A5:\n{A5}")
print(f"Собственные значения A5: {np.linalg.eigvals(A5)}")
print()

# 6-8. Те же собственные числа, умноженные на c = 0.5 (0 < c < 1)
c = 0.5
eigenvals_6 = [eigenvals_1[0] * c, eigenvals_1[1] * c]  # λ1,2 = -0.5
A6 = create_matrix_with_eigenvalues(eigenvals_6)
print(f"Система 6 - λ1,2 = -0.5 (c = {c})")
print(f"Матрица A6:\n{A6}")
print(f"Собственные значения A6: {np.linalg.eigvals(A6)}")
print()

eigenvals_7 = [eigenvals_3[0] * c, eigenvals_3[1] * c]  # λ1,2 = ±0.5i
A7 = create_matrix_with_eigenvalues(eigenvals_7)
print(f"Система 7 - λ1,2 = ±0.5i (c = {c})")
print(f"Матрица A7:\n{A7}")
print(f"Собственные значения A7: {np.linalg.eigvals(A7)}")
print()

eigenvals_8 = [eigenvals_5[0] * c, eigenvals_5[1] * c]  # λ1,2 = 0.5
A8 = create_matrix_with_eigenvalues(eigenvals_8)
print(f"Система 8 - λ1,2 = 0.5 (c = {c})")
print(f"Матрица A8:\n{A8}")
print(f"Собственные значения A8: {np.linalg.eigvals(A8)}")
print()

# 9-11. Те же собственные числа, умноженные на d = 1.5 (d > 1)
d = 1.5
eigenvals_9 = [eigenvals_1[0] * d, eigenvals_1[1] * d]  # λ1,2 = -1.5
A9 = create_matrix_with_eigenvalues(eigenvals_9)
print(f"Система 9 - λ1,2 = -1.5 (d = {d})")
print(f"Матрица A9:\n{A9}")
print(f"Собственные значения A9: {np.linalg.eigvals(A9)}")
print()

eigenvals_10 = [eigenvals_3[0] * d, eigenvals_3[1] * d]  # λ1,2 = ±1.5i
A10 = create_matrix_with_eigenvalues(eigenvals_10)
print(f"Система 10 - λ1,2 = ±1.5i (d = {d})")
print(f"Матрица A10:\n{A10}")
print(f"Собственные значения A10: {np.linalg.eigvals(A10)}")
print()

eigenvals_11 = [eigenvals_5[0] * d, eigenvals_5[1] * d]  # λ1,2 = 1.5
A11 = create_matrix_with_eigenvalues(eigenvals_11)
print(f"Система 11 - λ1,2 = 1.5 (d = {d})")
print(f"Матрица A11:\n{A11}")
print(f"Собственные значения A11: {np.linalg.eigvals(A11)}")
print()

# 12. λ1,2 = 0
eigenvals_12 = [0, 0]
A12 = create_matrix_with_eigenvalues(eigenvals_12)
print(f"Система 12 - λ1,2 = 0")
print(f"Матрица A12:\n{A12}")
print(f"Собственные значения A12: {np.linalg.eigvals(A12)}")
print()

# Строим графики для всех систем
systems = [
    (A1, "Система 1 - λ1,2 = -1"),
    (A2, "Система 2 - λ1,2 = -1/√2 ± 1/√2*i"),
    (A3, "Система 3 - λ1,2 = ±i"),
    (A4, "Система 4 - λ1,2 = 1/√2 ± 1/√2*i"),
    (A5, "Система 5 - λ1,2 = 1"),
    (A6, "Система 6 - λ1,2 = -0.5 (c = 0.5)"),
    (A7, "Система 7 - λ1,2 = ±0.5i (c = 0.5)"),
    (A8, "Система 8 - λ1,2 = 0.5 (c = 0.5)"),
    (A9, "Система 9 - λ1,2 = -1.5 (d = 1.5)"),
    (A10, "Система 10 - λ1,2 = ±1.5i (d = 1.5)"),
    (A11, "Система 11 - λ1,2 = 1.5 (d = 1.5)"),
    (A12, "Система 12 - λ1,2 = 0")
]

for A, name in systems:
    print(f"Строим графики для {name}...")
    plot_discrete_system(A, name)

print("\nВсе графики сохранены в папке ../images/task2/") 