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
c = 0.5          # амплитуда гармонической помехи
d = 10.0         # частота гармонической помехи

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
harmonic_noise = c * np.sin(d * t)
u = g + noise + harmonic_noise

# Вычисляем Фурье-образ
U = fftshift(fft(u))

# Создаем массив частот
V = 1/dt  # ширина диапазона частот
dv = 1/T  # шаг частоты
v = np.arange(-V/2, V/2, dv)
# Убеждаемся, что размер массива частот совпадает с размером Фурье-образа
if len(v) != N:
    v = np.linspace(-V/2, V/2, N, endpoint=False)

# Функция для фильтрации низких частот
def filter_low_frequencies(U, v, nu0):
    """
    Фильтрует низкие частоты, обнуляя Фурье-образ в окрестности ν = 0
    nu0: радиус окрестности нулевой частоты
    """
    U_filtered = U.copy()
    # Обнуляем частоты в окрестности ν = 0
    mask = np.abs(v) <= nu0
    U_filtered[mask] = 0
    return U_filtered

# Исследуем влияние различных радиусов окрестности
nu0_values = [0.5, 1.0, 2.0, 5.0]

plt.figure(figsize=(15, 10))

for i, nu0 in enumerate(nu0_values):
    # Применяем фильтр
    U_filtered = filter_low_frequencies(U, v, nu0)
    u_filtered = np.real(ifft(ifftshift(U_filtered)))
    
    # График 1: Сравнение сигналов во временной области
    plt.subplot(2, 2, i+1)
    
    # Показываем только окрестность интервала [t1, t2]
    mask_plot = (t >= t1-1) & (t <= t2+1)
    
    plt.plot(t[mask_plot], g[mask_plot], 'b-', linewidth=2, label='Исходный сигнал g(t)')
    plt.plot(t[mask_plot], u[mask_plot], 'r-', alpha=0.7, label='Зашумлённый сигнал u(t)')
    plt.plot(t[mask_plot], u_filtered[mask_plot], 'g-', linewidth=2, label=f'Отфильтрованный (ν₀={nu0})')
    
    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.title(f'Фильтрация низких частот, ν₀ = {nu0}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task1/low_freq_filter_time_domain.png', dpi=300, bbox_inches='tight')
plt.close()

# График 2: Сравнение Фурье-образов
plt.figure(figsize=(15, 10))

for i, nu0 in enumerate(nu0_values):
    U_filtered = filter_low_frequencies(U, v, nu0)
    
    plt.subplot(2, 2, i+1)
    
    # Показываем только положительные частоты для наглядности
    mask_freq = v >= 0
    
    plt.plot(v[mask_freq], np.abs(U[mask_freq]), 'r-', alpha=0.7, label='Исходный сигнал')
    plt.plot(v[mask_freq], np.abs(U_filtered[mask_freq]), 'g-', linewidth=2, label=f'Отфильтрованный (ν₀={nu0})')
    plt.axvspan(0, nu0, alpha=0.2, color='red', label=f'Обнулённая область [0, {nu0}]')
    
    plt.xlabel('Частота ν')
    plt.ylabel('|Ũ(ν)|')
    plt.title(f'Фурье-образы, ν₀ = {nu0}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 15)

plt.tight_layout()
plt.savefig('../images/task1/low_freq_filter_freq_domain.png', dpi=300, bbox_inches='tight')
plt.close()

# Анализ влияния на различные типы сигналов
plt.figure(figsize=(15, 10))

# Случай 1: Только случайный шум (b ≠ 0, c = 0)
c_val = 0.0
harmonic_noise_c0 = c_val * np.sin(d * t)
u_c0 = g + noise + harmonic_noise_c0
U_c0 = fftshift(fft(u_c0))

# Применяем фильтр
U_filtered_c0 = filter_low_frequencies(U_c0, v, 2.0)
u_filtered_c0 = np.real(ifft(ifftshift(U_filtered_c0)))

plt.subplot(2, 2, 1)
mask_plot = (t >= t1-1) & (t <= t2+1)
plt.plot(t[mask_plot], g[mask_plot], 'b-', linewidth=2, label='Исходный сигнал')
plt.plot(t[mask_plot], u_c0[mask_plot], 'r-', alpha=0.7, label='Зашумлённый (только случайный шум)')
plt.plot(t[mask_plot], u_filtered_c0[mask_plot], 'g-', linewidth=2, label='Отфильтрованный')
plt.xlabel('Время t')
plt.ylabel('Амплитуда')
plt.title('Только случайный шум (c=0)')
plt.legend()
plt.grid(True, alpha=0.3)

# Случай 2: Только гармоническая помеха (b = 0, c ≠ 0)
b_val = 0.0
noise_b0 = b_val * (np.random.rand(N) - 0.5)
u_b0 = g + noise_b0 + harmonic_noise
U_b0 = fftshift(fft(u_b0))

# Применяем фильтр
U_filtered_b0 = filter_low_frequencies(U_b0, v, 2.0)
u_filtered_b0 = np.real(ifft(ifftshift(U_filtered_b0)))

plt.subplot(2, 2, 2)
plt.plot(t[mask_plot], g[mask_plot], 'b-', linewidth=2, label='Исходный сигнал')
plt.plot(t[mask_plot], u_b0[mask_plot], 'r-', alpha=0.7, label='Зашумлённый (только гармоническая помеха)')
plt.plot(t[mask_plot], u_filtered_b0[mask_plot], 'g-', linewidth=2, label='Отфильтрованный')
plt.xlabel('Время t')
plt.ylabel('Амплитуда')
plt.title('Только гармоническая помеха (b=0)')
plt.legend()
plt.grid(True, alpha=0.3)

# Случай 3: Комбинированные помехи
plt.subplot(2, 2, 3)
plt.plot(t[mask_plot], g[mask_plot], 'b-', linewidth=2, label='Исходный сигнал')
plt.plot(t[mask_plot], u[mask_plot], 'r-', alpha=0.7, label='Зашумлённый (комбинированные помехи)')
U_filtered_combined = filter_low_frequencies(U, v, 2.0)
u_filtered_combined = np.real(ifft(ifftshift(U_filtered_combined)))
plt.plot(t[mask_plot], u_filtered_combined[mask_plot], 'g-', linewidth=2, label='Отфильтрованный')
plt.xlabel('Время t')
plt.ylabel('Амплитуда')
plt.title('Комбинированные помехи')
plt.legend()
plt.grid(True, alpha=0.3)

# Случай 4: Сравнение различных радиусов окрестности
plt.subplot(2, 2, 4)
plt.plot(t[mask_plot], g[mask_plot], 'b-', linewidth=2, label='Исходный сигнал')
plt.plot(t[mask_plot], u[mask_plot], 'r-', alpha=0.7, label='Зашумлённый сигнал')

colors = ['green', 'orange', 'purple', 'red']
for i, nu0 in enumerate([0.5, 1.0, 2.0, 5.0]):
    U_filtered_nu = filter_low_frequencies(U, v, nu0)
    u_filtered_nu = np.real(ifft(ifftshift(U_filtered_nu)))
    plt.plot(t[mask_plot], u_filtered_nu[mask_plot], color=colors[i], 
             linewidth=2, label=f'Отфильтрованный (ν₀={nu0})')

plt.xlabel('Время t')
plt.ylabel('Амплитуда')
plt.title('Сравнение различных ν₀')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task1/low_freq_filter_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Количественный анализ эффективности фильтрации
print("Анализ эффективности фильтрации низких частот:")
print("=" * 50)

for nu0 in nu0_values:
    U_filtered = filter_low_frequencies(U, v, nu0)
    u_filtered = np.real(ifft(ifftshift(U_filtered)))
    
    # Вычисляем среднеквадратичную ошибку
    mse = np.mean((g - u_filtered)**2)
    # Вычисляем корреляцию с исходным сигналом
    correlation = np.corrcoef(g, u_filtered)[0, 1]
    
    print(f"ν₀ = {nu0}:")
    print(f"  Среднеквадратичная ошибка: {mse:.6f}")
    print(f"  Корреляция с исходным сигналом: {correlation:.6f}")
    print()

print("Фильтрация низких частот завершена!")
print(f"Параметры: a={a}, t1={t1}, t2={t2}, T={T}, dt={dt}")
print(f"Параметры помех: b={b}, c={c}, d={d}")
print(f"Исследованные радиусы окрестности: {nu0_values}") 