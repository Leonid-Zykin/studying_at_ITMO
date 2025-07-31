import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# Создаем папку для изображений если её нет
os.makedirs('../images/task3', exist_ok=True)

def oscillator_system(x, t, a, b):
    """Система осциллятора: x1' = x2, x2' = a*x1 + b*x2"""
    x1, x2 = x
    dx1_dt = x2
    dx2_dt = a * x1 + b * x2
    return [dx1_dt, dx2_dt]

def analyze_oscillator(a, b, case_name, t_span=np.linspace(0, 10, 1000)):
    """Анализирует осциллятор с заданными параметрами"""
    
    # Начальные условия
    x0_1 = np.array([1, 0])  # Начальное смещение, нулевая скорость
    x0_2 = np.array([0, 1])  # Нулевое смещение, начальная скорость
    x0_3 = np.array([1, 1])  # Начальное смещение и скорость
    
    initial_conditions = [x0_1, x0_2, x0_3]
    colors = ['red', 'blue', 'green']
    labels = ['x0 = [1, 0]', 'x0 = [0, 1]', 'x0 = [1, 1]']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Решаем систему для каждого начального условия
    for i, (x0, color, label) in enumerate(zip(initial_conditions, colors, labels)):
        solution = odeint(oscillator_system, x0, t_span, args=(a, b))
        
        # График x1(t) - смещение
        ax1.plot(t_span, solution[:, 0], color=color, label=f'{label}: x1(t)')
        
        # График x2(t) - скорость
        ax2.plot(t_span, solution[:, 1], color=color, label=f'{label}: x2(t)')
        
        # Фазовый портрет
        ax3.plot(solution[:, 0], solution[:, 1], color=color, label=label)
        
        # Начальная точка
        ax3.plot(solution[0, 0], solution[0, 1], 'o', color=color, markersize=8)
    
    # Настройка графиков
    ax1.set_xlabel('t')
    ax1.set_ylabel('x1(t) - смещение')
    ax1.set_title(f'{case_name} - График смещения x1(t)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('t')
    ax2.set_ylabel('x2(t) - скорость')
    ax2.set_title(f'{case_name} - График скорости x2(t)')
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_xlabel('x1 - смещение')
    ax3.set_ylabel('x2 - скорость')
    ax3.set_title(f'{case_name} - Фазовый портрет')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # Анализ устойчивости
    # Матрица системы: A = [[0, 1], [a, b]]
    A = np.array([[0, 1], [a, b]])
    eigenvals = np.linalg.eigvals(A)
    
    # График собственных значений
    for i, eigenval in enumerate(eigenvals):
        ax4.plot(eigenval.real, eigenval.imag, 'o', markersize=10, 
                label=f'λ{i+1} = {eigenval:.3f}')
    
    ax4.set_xlabel('Re(λ)')
    ax4.set_ylabel('Im(λ)')
    ax4.set_title(f'{case_name} - Собственные значения')
    ax4.legend()
    ax4.grid(True)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Определяем тип устойчивости
    real_parts = eigenvals.real
    if np.all(real_parts < 0):
        stability = "Асимптотически устойчива"
    elif np.any(real_parts > 0):
        stability = "Неустойчива"
    else:
        stability = "Нейтрально устойчива"
    
    ax4.text(0.02, 0.98, f'Устойчивость: {stability}', 
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'../images/task3/{case_name.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return eigenvals, stability

# Анализируем все 4 случая

print("=== АНАЛИЗ ОСЦИЛЛЯТОРА ===\n")

# Случай 1: a < 0, b = 0
print("Случай 1: a < 0, b = 0")
print("Физическая интерпретация: Гармонический осциллятор без затухания")
print("x1 - смещение от положения равновесия")
print("x2 - скорость")
print("a - коэффициент упругости (отрицательный)")
print("b - коэффициент затухания (равен нулю)")
print()

a1, b1 = -1, 0
eigenvals1, stability1 = analyze_oscillator(a1, b1, "Случай 1 - a < 0, b = 0 (Гармонический осциллятор)")
print(f"Собственные значения: {eigenvals1}")
print(f"Устойчивость: {stability1}")
print()

# Случай 2: a < 0, b < 0
print("Случай 2: a < 0, b < 0")
print("Физическая интерпретация: Затухающий гармонический осциллятор")
print("x1 - смещение от положения равновесия")
print("x2 - скорость")
print("a - коэффициент упругости (отрицательный)")
print("b - коэффициент затухания (отрицательный)")
print()

a2, b2 = -1, -0.5
eigenvals2, stability2 = analyze_oscillator(a2, b2, "Случай 2 - a < 0, b < 0 (Затухающий осциллятор)")
print(f"Собственные значения: {eigenvals2}")
print(f"Устойчивость: {stability2}")
print()

# Случай 3: a > 0, b = 0
print("Случай 3: a > 0, b = 0")
print("Физическая интерпретация: Неустойчивый осциллятор (например, маятник в перевернутом положении)")
print("x1 - смещение от неустойчивого положения равновесия")
print("x2 - скорость")
print("a - коэффициент упругости (положительный - неустойчивость)")
print("b - коэффициент затухания (равен нулю)")
print()

a3, b3 = 1, 0
eigenvals3, stability3 = analyze_oscillator(a3, b3, "Случай 3 - a > 0, b = 0 (Неустойчивый осциллятор)")
print(f"Собственные значения: {eigenvals3}")
print(f"Устойчивость: {stability3}")
print()

# Случай 4: a > 0, b < 0
print("Случай 4: a > 0, b < 0")
print("Физическая интерпретация: Неустойчивый осциллятор с затуханием")
print("x1 - смещение от неустойчивого положения равновесия")
print("x2 - скорость")
print("a - коэффициент упругости (положительный - неустойчивость)")
print("b - коэффициент затухания (отрицательный)")
print()

a4, b4 = 1, -0.5
eigenvals4, stability4 = analyze_oscillator(a4, b4, "Случай 4 - a > 0, b < 0 (Неустойчивый осциллятор с затуханием)")
print(f"Собственные значения: {eigenvals4}")
print(f"Устойчивость: {stability4}")
print()

# Сводная таблица результатов
print("=== СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===")
print(f"{'Случай':<50} {'Собственные значения':<30} {'Устойчивость':<25}")
print("-" * 105)
print(f"{'1. a < 0, b = 0 (Гармонический осциллятор)':<50} {str(eigenvals1):<30} {stability1:<25}")
print(f"{'2. a < 0, b < 0 (Затухающий осциллятор)':<50} {str(eigenvals2):<30} {stability2:<25}")
print(f"{'3. a > 0, b = 0 (Неустойчивый осциллятор)':<50} {str(eigenvals3):<30} {stability3:<25}")
print(f"{'4. a > 0, b < 0 (Неустойчивый с затуханием)':<50} {str(eigenvals4):<30} {stability4:<25}")

print("\nВсе графики сохранены в папке ../images/task3/") 