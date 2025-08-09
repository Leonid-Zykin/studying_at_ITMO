import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy import signal
import os

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task2', exist_ok=True)

# Параметры эксперимента
dt = 0.01
t = np.arange(-20, 20 + dt, dt)
N = len(t)

# Создаем прямоугольный импульс g(t)
a = 1.0
t1 = -2.0
t2 = 2.0
g = np.zeros_like(t)
mask = (t >= t1) & (t <= t2)
g[mask] = a

# Создаем зашумлённый сигнал
np.random.seed(42)
b = 0.5  # амплитуда случайного шума
c = 0.0  # амплитуда гармонической помехи (нулевая для этого задания)
noise = b * (np.random.rand(N) - 0.5)
u = g + noise

# Функция для создания фильтра первого порядка
def create_first_order_filter(T):
    """Создает фильтр первого порядка с передаточной функцией W(p) = 1/(T*p + 1)"""
    # В дискретном времени: W(z) = 1/(T*(z-1)/(dt*z) + 1) = z/(T*(z-1)/dt + z)
    # Приводим к виду: W(z) = (dt*z)/(T*(z-1) + dt*z) = (dt*z)/((T+dt)*z - T)
    # Нормализуем: W(z) = (dt/(T+dt)*z)/(z - T/(T+dt))
    
    alpha = dt / (T + dt)
    beta = T / (T + dt)
    
    # Коэффициенты фильтра: y[n] = alpha * x[n] + beta * y[n-1]
    return alpha, beta

# Функция для применения фильтра
def apply_filter(x, alpha, beta):
    """Применяет фильтр первого порядка к сигналу x"""
    y = np.zeros_like(x)
    y[0] = alpha * x[0]  # начальное условие
    
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + beta * y[i-1]
    
    return y

# Исследуем различные значения постоянной времени
T_values = [0.1, 0.5, 1.0, 2.0]

plt.figure(figsize=(15, 10))

for i, T in enumerate(T_values):
    # Создаем фильтр
    alpha, beta = create_first_order_filter(T)
    
    # Применяем фильтр
    u_filtered = apply_filter(u, alpha, beta)
    
    # График во временной области
    plt.subplot(2, 2, i+1)
    
    # Показываем только окрестность интервала [t1, t2]
    mask_plot = (t >= t1-1) & (t <= t2+1)
    
    plt.plot(t[mask_plot], g[mask_plot], 'g-', linewidth=2, label='Исходный сигнал g(t)')
    plt.plot(t[mask_plot], u[mask_plot], 'r-', alpha=0.7, linewidth=1, label='Зашумлённый сигнал u(t)')
    plt.plot(t[mask_plot], u_filtered[mask_plot], 'b-', linewidth=2, label=f'Отфильтрованный (T={T})')
    
    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.title(f'Фильтр первого порядка, T = {T}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task2/first_order_filter_time_domain.png', dpi=300, bbox_inches='tight')
plt.close()

# График 2: Фурье-образы
plt.figure(figsize=(15, 10))

for i, T in enumerate(T_values):
    alpha, beta = create_first_order_filter(T)
    u_filtered = apply_filter(u, alpha, beta)
    
    # Вычисляем Фурье-образы
    U = fftshift(fft(u))
    U_filtered = fftshift(fft(u_filtered))
    
    # Создаем массив частот
    T_total = t[-1] - t[0]
    df = 1 / T_total
    f = np.linspace(-N//2, N//2, N, endpoint=False) * df
    
    plt.subplot(2, 2, i+1)
    
    # Показываем только положительные частоты до 10 Гц
    mask_freq = (f >= 0) & (f <= 10)
    
    plt.plot(f[mask_freq], np.abs(U[mask_freq]), 'r-', alpha=0.7, linewidth=1, label='Исходный сигнал')
    plt.plot(f[mask_freq], np.abs(U_filtered[mask_freq]), 'g-', linewidth=2, label=f'Отфильтрованный (T={T})')
    
    plt.xlabel('Частота f')
    plt.ylabel('|U(f)|')
    plt.title(f'Фурье-образы, T = {T}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task2/first_order_filter_freq_domain.png', dpi=300, bbox_inches='tight')
plt.close()

# График 3: АЧХ фильтров
plt.figure(figsize=(15, 10))

for i, T in enumerate(T_values):
    # Теоретическая АЧХ фильтра первого порядка
    f_theoretical = np.logspace(-2, 2, 1000)
    omega = 2 * np.pi * f_theoretical
    
    # АЧХ: |W(jω)| = 1/√(1 + (ωT)²)
    magnitude = 1 / np.sqrt(1 + (omega * T)**2)
    
    plt.subplot(2, 2, i+1)
    plt.plot(f_theoretical, magnitude, 'b-', linewidth=2, label=f'АЧХ (T={T})')
    
    # Частота среза: ω_c = 1/T
    f_cutoff = 1 / (2 * np.pi * T)
    plt.axvline(x=f_cutoff, color='red', linestyle='--', alpha=0.7, label=f'Частота среза {f_cutoff:.3f} Гц')
    
    plt.xlabel('Частота f (Гц)')
    plt.ylabel('|W(jω)|')
    plt.title(f'АЧХ фильтра первого порядка, T = {T}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 10)  # Показываем только до 10 Гц для наглядности
    plt.ylim(0, 1.1)

plt.tight_layout()
plt.savefig('../images/task2/first_order_filter_frequency_response.png', dpi=300, bbox_inches='tight')
plt.close()

# Анализ влияния параметра b на эффективность фильтрации
plt.figure(figsize=(15, 10))

T_fixed = 1.0  # фиксированная постоянная времени
b_values = [0.1, 0.3, 0.5, 0.7]

for i, b_val in enumerate(b_values):
    # Создаем зашумлённый сигнал с новым значением b
    noise_b = b_val * (np.random.rand(N) - 0.5)
    u_b = g + noise_b
    
    # Применяем фильтр
    alpha, beta = create_first_order_filter(T_fixed)
    u_filtered_b = apply_filter(u_b, alpha, beta)
    
    plt.subplot(2, 2, i+1)
    mask_plot = (t >= t1-1) & (t <= t2+1)
    
    plt.plot(t[mask_plot], g[mask_plot], 'g-', linewidth=2, label='Исходный сигнал')
    plt.plot(t[mask_plot], u_b[mask_plot], 'r-', alpha=0.7, linewidth=1, label=f'Зашумлённый (b={b_val})')
    plt.plot(t[mask_plot], u_filtered_b[mask_plot], 'b-', linewidth=2, label='Отфильтрованный')
    
    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.title(f'Влияние параметра b, b = {b_val}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task2/first_order_filter_b_influence.png', dpi=300, bbox_inches='tight')
plt.close()

# Количественный анализ эффективности фильтрации
print("Анализ фильтра первого порядка:")
print("=" * 50)

for T in T_values:
    alpha, beta = create_first_order_filter(T)
    u_filtered = apply_filter(u, alpha, beta)
    
    # Вычисляем среднеквадратичную ошибку
    mse = np.mean((g - u_filtered)**2)
    # Вычисляем корреляцию с исходным сигналом
    correlation = np.corrcoef(g, u_filtered)[0, 1]
    
    # Частота среза
    f_cutoff = 1 / (2 * np.pi * T)
    
    print(f"T = {T}:")
    print(f"  Частота среза: {f_cutoff:.3f} Гц")
    print(f"  Среднеквадратичная ошибка: {mse:.6f}")
    print(f"  Корреляция с исходным сигналом: {correlation:.6f}")
    print()

print(f"Параметры: a={a}, t1={t1}, t2={t2}, b={b}, c={c}")
print(f"Исследованные значения T: {T_values}") 