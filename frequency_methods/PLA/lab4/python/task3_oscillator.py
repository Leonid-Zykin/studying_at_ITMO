import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# Создаем папку для сохранения изображений
os.makedirs('../images/task3', exist_ok=True)

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Функция для решения системы осциллятора
def oscillator_ode(x, t, a, b):
    """Система осциллятора: dx1/dt = x2, dx2/dt = ax1 + bx2"""
    return [x[1], a * x[0] + b * x[1]]

# Функция для построения графиков осциллятора
def plot_oscillator(a, b, title, filename_base, physical_interpretation):
    """Построение графиков для осциллятора с заданными параметрами"""
    
    # Время интегрирования
    t = np.linspace(0, 20, 1000)
    
    # Начальные условия
    x0_list = [
        [1, 0],    # смещение без начальной скорости
        [0, 1],    # начальная скорость без смещения
        [1, 1],    # смещение и начальная скорость
        [2, -1]    # произвольные начальные условия
    ]
    
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['x(0)=[1,0]', 'x(0)=[0,1]', 'x(0)=[1,1]', 'x(0)=[2,-1]']
    
    # График 1: x1(t) и x2(t)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for i, x0 in enumerate(x0_list):
        solution = odeint(oscillator_ode, x0, t, args=(a, b))
        plt.plot(t, solution[:, 0], color=colors[i], label=f'x1(t), {labels[i]}')
    plt.xlabel('Время t')
    plt.ylabel('x1(t)')
    plt.title(f'{title}\nx1(t) - Положение')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    for i, x0 in enumerate(x0_list):
        solution = odeint(oscillator_ode, x0, t, args=(a, b))
        plt.plot(t, solution[:, 1], color=colors[i], label=f'x2(t), {labels[i]}')
    plt.xlabel('Время t')
    plt.ylabel('x2(t)')
    plt.title(f'{title}\nx2(t) - Скорость')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 3: Фазовая плоскость
    plt.subplot(2, 2, 3)
    for i, x0 in enumerate(x0_list):
        solution = odeint(oscillator_ode, x0, t, args=(a, b))
        plt.plot(solution[:, 0], solution[:, 1], color=colors[i], 
                label=f'{labels[i]}', linewidth=2)
        # Отмечаем начальную точку
        plt.plot(solution[0, 0], solution[0, 1], 'o', color=colors[i], markersize=8)
    
    plt.xlabel('x1 (Положение)')
    plt.ylabel('x2 (Скорость)')
    plt.title(f'{title}\nФазовая плоскость')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # График 4: Энергия системы
    plt.subplot(2, 2, 4)
    for i, x0 in enumerate(x0_list):
        solution = odeint(oscillator_ode, x0, t, args=(a, b))
        # Кинетическая энергия: E_k = (1/2) * m * v^2 = (1/2) * x2^2
        # Потенциальная энергия: E_p = (1/2) * k * x^2 = (1/2) * |a| * x1^2
        kinetic_energy = 0.5 * solution[:, 1]**2
        potential_energy = 0.5 * abs(a) * solution[:, 0]**2
        total_energy = kinetic_energy + potential_energy
        
        plt.plot(t, total_energy, color=colors[i], label=f'Полная энергия, {labels[i]}')
    
    plt.xlabel('Время t')
    plt.ylabel('Энергия')
    plt.title(f'{title}\nПолная энергия системы')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../images/task3/{filename_base}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Анализ устойчивости
    eigenvalues = np.linalg.eigvals([[0, 1], [a, b]])
    print(f"\n{title}")
    print(f"Параметры: a = {a}, b = {b}")
    print(f"Собственные числа: {eigenvalues}")
    print(f"Реальные части: {np.real(eigenvalues)}")
    print(f"Модули: {np.abs(eigenvalues)}")
    
    # Определение типа устойчивости
    real_parts = np.real(eigenvalues)
    if np.all(real_parts < 0):
        stability = "Асимптотически устойчива"
    elif np.any(real_parts > 0):
        stability = "Неустойчива"
    else:
        stability = "Нейтрально устойчива"
    
    print(f"Тип устойчивости: {stability}")
    print(f"Физическая интерпретация: {physical_interpretation}")
    print()

# Случай 1: a < 0, b = 0 (гармонический осциллятор)
a1, b1 = -1, 0
physical1 = "Гармонический осциллятор (пружина без трения):\n" + \
           "x1 - смещение от положения равновесия\n" + \
           "x2 - скорость\n" + \
           "a < 0 - коэффициент упругости (отрицательный для возвращающей силы)\n" + \
           "b = 0 - отсутствие трения"

plot_oscillator(a1, b1, "Случай 1: a < 0, b = 0\nГармонический осциллятор", 
               "oscillator_case1_harmonic", physical1)

# Случай 2: a < 0, b < 0 (затухающий осциллятор)
a2, b2 = -1, -0.5
physical2 = "Затухающий осциллятор (пружина с трением):\n" + \
           "x1 - смещение от положения равновесия\n" + \
           "x2 - скорость\n" + \
           "a < 0 - коэффициент упругости\n" + \
           "b < 0 - коэффициент трения (отрицательный для затухания)"

plot_oscillator(a2, b2, "Случай 2: a < 0, b < 0\nЗатухающий осциллятор", 
               "oscillator_case2_damped", physical2)

# Случай 3: a > 0, b = 0 (неустойчивый осциллятор)
a3, b3 = 1, 0
physical3 = "Неустойчивый осциллятор (перевернутый маятник):\n" + \
           "x1 - угол отклонения от неустойчивого равновесия\n" + \
           "x2 - угловая скорость\n" + \
           "a > 0 - коэффициент неустойчивости (положительный для отталкивающей силы)\n" + \
           "b = 0 - отсутствие трения"

plot_oscillator(a3, b3, "Случай 3: a > 0, b = 0\nНеустойчивый осциллятор", 
               "oscillator_case3_unstable", physical3)

# Случай 4: a > 0, b < 0 (неустойчивый осциллятор с затуханием)
a4, b4 = 1, -0.5
physical4 = "Неустойчивый осциллятор с затуханием:\n" + \
           "x1 - отклонение от неустойчивого равновесия\n" + \
           "x2 - скорость\n" + \
           "a > 0 - коэффициент неустойчивости\n" + \
           "b < 0 - коэффициент трения (может стабилизировать систему)"

plot_oscillator(a4, b4, "Случай 4: a > 0, b < 0\nНеустойчивый осциллятор с затуханием", 
               "oscillator_case4_unstable_damped", physical4)

# Сравнительный анализ всех случаев
plt.figure(figsize=(15, 10))

cases = [
    (a1, b1, "Случай 1: a<0, b=0", "Гармонический"),
    (a2, b2, "Случай 2: a<0, b<0", "Затухающий"),
    (a3, b3, "Случай 3: a>0, b=0", "Неустойчивый"),
    (a4, b4, "Случай 4: a>0, b<0", "Неустойчивый с затуханием")
]

colors = ['blue', 'green', 'red', 'orange']

for i, (a, b, title, label) in enumerate(cases):
    t = np.linspace(0, 10, 500)
    x0 = [1, 0]  # одинаковые начальные условия для сравнения
    solution = odeint(oscillator_ode, x0, t, args=(a, b))
    
    plt.subplot(2, 2, i+1)
    plt.plot(t, solution[:, 0], color=colors[i], linewidth=2, label=f'x1(t)')
    plt.plot(t, solution[:, 1], color=colors[i], linestyle='--', linewidth=2, label=f'x2(t)')
    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.title(f'{title}\n{label}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task3/oscillator_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Анализ осциллятора завершен!")
print("Все графики сохранены в папке images/task3/") 