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

# Создаем зашумлённый сигнал с гармонической помехой
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

# Функция для фильтрации специфических частот
def filter_specific_frequencies(U, v, freq_ranges):
    """
    Фильтрует специфические частоты
    freq_ranges: список кортежей [(freq1_low, freq1_high), (freq2_low, freq2_high), ...]
    """
    U_filtered = U.copy()
    
    for freq_low, freq_high in freq_ranges:
        # Обнуляем частоты в заданном диапазоне
        mask = (np.abs(v) >= freq_low) & (np.abs(v) <= freq_high)
        U_filtered[mask] = 0
    
    return U_filtered

# Исследуем различные стратегии фильтрации
strategies = [
    {
        'name': 'Фильтрация высоких частот',
        'freq_ranges': [(5.0, 50.0)],  # убираем высокие частоты
        'color': 'green'
    },
    {
        'name': 'Фильтрация гармонической помехи',
        'freq_ranges': [(d-0.5, d+0.5)],  # убираем частоту помехи
        'color': 'orange'
    },
    {
        'name': 'Комбинированная фильтрация',
        'freq_ranges': [(5.0, 50.0), (d-0.5, d+0.5)],  # убираем и высокие частоты, и помеху
        'color': 'purple'
    },
    {
        'name': 'Селективная фильтрация',
        'freq_ranges': [(0.5, 2.0), (d-0.5, d+0.5), (15.0, 25.0)],  # убираем несколько диапазонов
        'color': 'red'
    }
]

# График 1: Сравнение сигналов во временной области
plt.figure(figsize=(15, 10))

for i, strategy in enumerate(strategies):
    # Применяем фильтр
    U_filtered = filter_specific_frequencies(U, v, strategy['freq_ranges'])
    u_filtered = np.real(ifft(ifftshift(U_filtered)))
    
    plt.subplot(2, 2, i+1)
    
    # Показываем только окрестность интервала [t1, t2]
    mask_plot = (t >= t1-1) & (t <= t2+1)
    
    plt.plot(t[mask_plot], g[mask_plot], 'b-', linewidth=2, label='Исходный сигнал g(t)')
    plt.plot(t[mask_plot], u[mask_plot], 'r-', alpha=0.7, label='Зашумлённый сигнал u(t)')
    plt.plot(t[mask_plot], u_filtered[mask_plot], color=strategy['color'], 
             linewidth=2, label=f'Отфильтрованный ({strategy["name"]})')
    
    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.title(f'{strategy["name"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task1/specific_freq_filter_time_domain.png', dpi=300, bbox_inches='tight')
plt.close()

# График 2: Сравнение Фурье-образов
plt.figure(figsize=(15, 10))

for i, strategy in enumerate(strategies):
    U_filtered = filter_specific_frequencies(U, v, strategy['freq_ranges'])
    
    plt.subplot(2, 2, i+1)
    
    # Показываем только положительные частоты для наглядности
    mask_freq = v >= 0
    
    plt.plot(v[mask_freq], np.abs(U[mask_freq]), 'r-', alpha=0.7, label='Исходный сигнал')
    plt.plot(v[mask_freq], np.abs(U_filtered[mask_freq]), color=strategy['color'], 
             linewidth=2, label=f'Отфильтрованный')
    
    # Отмечаем диапазоны фильтрации
    for freq_low, freq_high in strategy['freq_ranges']:
        plt.axvspan(freq_low, freq_high, alpha=0.2, color='red', label=f'Диапазон фильтрации [{freq_low}, {freq_high}]')
    
    plt.xlabel('Частота ν')
    plt.ylabel('|Ũ(ν)|')
    plt.title(f'{strategy["name"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 20)

plt.tight_layout()
plt.savefig('../images/task1/specific_freq_filter_freq_domain.png', dpi=300, bbox_inches='tight')
plt.close()

# Исследование влияния параметров b, c, d
plt.figure(figsize=(15, 10))

# Случай 1: b = 0 (только гармоническая помеха)
b_val = 0.0
noise_b0 = b_val * (np.random.rand(N) - 0.5)
u_b0 = g + noise_b0 + harmonic_noise
U_b0 = fftshift(fft(u_b0))

# Применяем фильтр для гармонической помехи
U_filtered_b0 = filter_specific_frequencies(U_b0, v, [(d-0.5, d+0.5)])
u_filtered_b0 = np.real(ifft(ifftshift(U_filtered_b0)))

plt.subplot(2, 2, 1)
mask_plot = (t >= t1-1) & (t <= t2+1)
plt.plot(t[mask_plot], g[mask_plot], 'b-', linewidth=2, label='Исходный сигнал')
plt.plot(t[mask_plot], u_b0[mask_plot], 'r-', alpha=0.7, label=f'Зашумлённый (b={b_val})')
plt.plot(t[mask_plot], u_filtered_b0[mask_plot], 'g-', linewidth=2, label='Отфильтрованный')
plt.xlabel('Время t')
plt.ylabel('Амплитуда')
plt.title(f'Только гармоническая помеха (b={b_val})')
plt.legend()
plt.grid(True, alpha=0.3)

# Случай 2: Различные частоты гармонической помехи
d_values = [5.0, 10.0, 15.0, 20.0]
for i, d_val in enumerate(d_values[1:], 2):
    harmonic_noise_d = c * np.sin(d_val * t)
    u_d = g + noise + harmonic_noise_d
    U_d = fftshift(fft(u_d))
    
    # Применяем фильтр
    U_filtered_d = filter_specific_frequencies(U_d, v, [(d_val-0.5, d_val+0.5)])
    u_filtered_d = np.real(ifft(ifftshift(U_filtered_d)))
    
    plt.subplot(2, 2, i)
    plt.plot(t[mask_plot], g[mask_plot], 'b-', linewidth=2, label='Исходный сигнал')
    plt.plot(t[mask_plot], u_d[mask_plot], 'r-', alpha=0.7, label=f'Зашумлённый (d={d_val})')
    plt.plot(t[mask_plot], u_filtered_d[mask_plot], 'g-', linewidth=2, label='Отфильтрованный')
    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.title(f'Частота помехи d={d_val}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task1/specific_freq_filter_parameters.png', dpi=300, bbox_inches='tight')
plt.close()

print("Фильтрация специфических частот завершена!")
print(f"Параметры: a={a}, t1={t1}, t2={t2}, T={T}, dt={dt}")
print(f"Параметры помех: b={b}, c={c}, d={d}")
print(f"Исследованные стратегии фильтрации: {[s['name'] for s in strategies]}") 