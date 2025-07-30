import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
import os

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task1', exist_ok=True)

# Параметры эксперимента
a = 1.0          # амплитуда исходного сигнала
t1 = -2.0        # левая граница
t2 = 2.0         # правая граница
T = 20.0         # большой интервал времени
dt = 0.01        # шаг дискретизации
b = 0.3          # амплитуда случайного шума
c = 0.0          # амплитуда гармонической помехи (нулевая для этого задания)

# Создаем массив времени
t = np.arange(-T/2, T/2 + dt, dt)
N = len(t)

# Создаем исходную функцию g(t)
g = np.zeros_like(t)
mask = (t >= t1) & (t <= t2)
g[mask] = a

# Создаем зашумлённый сигнал
np.random.seed(42)  # для воспроизводимости результатов
noise = b * (np.random.rand(N) - 0.5)
u = g + noise

# Вычисляем Фурье-образ
U = fftshift(fft(u))

# Создаем массив частот
V = 1/dt  # ширина диапазона частот
dv = 1/T  # шаг частоты
v = np.arange(-V/2, V/2, dv)
# Убеждаемся, что размер массива частот совпадает с размером Фурье-образа
if len(v) != N:
    v = np.linspace(-V/2, V/2, N, endpoint=False)

# Функция для фильтрации высоких частот
def filter_high_frequencies(U, v, nu0):
    """Фильтрует высокие частоты, оставляя только диапазон [-nu0, nu0]"""
    U_filtered = U.copy()
    # Обнуляем частоты вне диапазона [-nu0, nu0]
    mask = (np.abs(v) > nu0)
    U_filtered[mask] = 0
    return U_filtered

# Исследуем влияние различных частот среза
nu0_values = [1.0, 2.0, 5.0, 10.0]

plt.figure(figsize=(15, 10))

for i, nu0 in enumerate(nu0_values):
    # Применяем фильтр
    U_filtered = filter_high_frequencies(U, v, nu0)
    
    # Восстанавливаем сигнал
    u_filtered = np.real(ifft(ifftshift(U_filtered)))
    
    # График 1: Сравнение сигналов во временной области
    plt.subplot(2, 2, i+1)
    
    # Показываем только окрестность интервала [t1, t2]
    mask_plot = (t >= t1-1) & (t <= t2+1)
    
    plt.plot(t[mask_plot], g[mask_plot], 'b-', linewidth=2, label='Исходный сигнал g(t)')
    plt.plot(t[mask_plot], u[mask_plot], 'r-', alpha=0.7, label='Зашумлённый сигнал u(t)')
    plt.plot(t[mask_plot], u_filtered[mask_plot], 'g-', linewidth=2, label=f'Отфильтрованный сигнал (ν₀={nu0})')
    
    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.title(f'Фильтрация высоких частот, ν₀ = {nu0}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task1/high_freq_filter_time_domain.png', dpi=300, bbox_inches='tight')
plt.close()

# График 2: Сравнение Фурье-образов
plt.figure(figsize=(15, 10))

for i, nu0 in enumerate(nu0_values):
    U_filtered = filter_high_frequencies(U, v, nu0)
    
    plt.subplot(2, 2, i+1)
    
    # Показываем только положительные частоты для наглядности
    mask_freq = v >= 0
    
    plt.plot(v[mask_freq], np.abs(U[mask_freq]), 'r-', alpha=0.7, label='Исходный сигнал')
    plt.plot(v[mask_freq], np.abs(U_filtered[mask_freq]), 'g-', linewidth=2, label=f'Отфильтрованный (ν₀={nu0})')
    plt.axvline(x=nu0, color='k', linestyle='--', alpha=0.5, label=f'Частота среза ν₀={nu0}')
    
    plt.xlabel('Частота ν')
    plt.ylabel('|Ũ(ν)|')
    plt.title(f'Фурье-образы, ν₀ = {nu0}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 15)

plt.tight_layout()
plt.savefig('../images/task1/high_freq_filter_freq_domain.png', dpi=300, bbox_inches='tight')
plt.close()

# Анализ влияния параметра b на эффективность фильтрации
b_values = [0.1, 0.3, 0.5, 0.7]
nu0 = 2.0  # фиксированная частота среза

plt.figure(figsize=(15, 10))

for i, b_val in enumerate(b_values):
    # Создаем зашумлённый сигнал с новым значением b
    noise = b_val * (np.random.rand(N) - 0.5)
    u_b = g + noise
    
    # Вычисляем Фурье-образ
    U_b = fftshift(fft(u_b))
    
    # Применяем фильтр
    U_filtered_b = filter_high_frequencies(U_b, v, nu0)
    u_filtered_b = np.real(ifft(ifftshift(U_filtered_b)))
    
    # График во временной области
    plt.subplot(2, 2, i+1)
    mask_plot = (t >= t1-1) & (t <= t2+1)
    
    plt.plot(t[mask_plot], g[mask_plot], 'b-', linewidth=2, label='Исходный сигнал')
    plt.plot(t[mask_plot], u_b[mask_plot], 'r-', alpha=0.7, label=f'Зашумлённый (b={b_val})')
    plt.plot(t[mask_plot], u_filtered_b[mask_plot], 'g-', linewidth=2, label='Отфильтрованный')
    
    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.title(f'Влияние параметра b, b = {b_val}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task1/high_freq_filter_b_influence.png', dpi=300, bbox_inches='tight')
plt.close()

print("Фильтрация высоких частот завершена!")
print(f"Параметры: a={a}, t1={t1}, t2={t2}, T={T}, dt={dt}")
print(f"Исследованные частоты среза: {nu0_values}")
print(f"Исследованные значения параметра b: {b_values}") 