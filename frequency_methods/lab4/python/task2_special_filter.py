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
b = 0.0  # амплитуда случайного шума (нулевая для этого задания)
c = 0.5  # амплитуда гармонической помехи

# Исследуем различные частоты помехи
d_values = [5, 10, 15, 20]  # частоты помехи в Гц

# Функция для создания специального фильтра второго порядка
def create_special_filter(T1, T2, T3):
    """Создает специальный фильтр с передаточной функцией W(p) = (T1*p + 1)^2 / ((T2*p + 1)(T3*p + 1))"""
    
    # В дискретном времени используем билинейное преобразование
    # p = 2*(z-1)/(dt*(z+1))
    
    # Коэффициенты числителя: (T1*p + 1)^2 = T1^2*p^2 + 2*T1*p + 1
    # Коэффициенты знаменателя: (T2*p + 1)(T3*p + 1) = T2*T3*p^2 + (T2 + T3)*p + 1
    
    # После билинейного преобразования получаем:
    # W(z) = (b0*z^2 + b1*z + b2) / (a0*z^2 + a1*z + a2)
    
    # Коэффициенты числителя
    b0 = (4*T1**2 + 4*T1*dt + dt**2) / (4*T2*T3 + 2*(T2 + T3)*dt + dt**2)
    b1 = (8*T1**2 - 2*dt**2) / (4*T2*T3 + 2*(T2 + T3)*dt + dt**2)
    b2 = (4*T1**2 - 4*T1*dt + dt**2) / (4*T2*T3 + 2*(T2 + T3)*dt + dt**2)
    
    # Коэффициенты знаменателя
    a0 = 1.0
    a1 = (8*T2*T3 - 2*dt**2) / (4*T2*T3 + 2*(T2 + T3)*dt + dt**2)
    a2 = (4*T2*T3 - 2*(T2 + T3)*dt + dt**2) / (4*T2*T3 + 2*(T2 + T3)*dt + dt**2)
    
    return [b0, b1, b2], [a0, a1, a2]

# Функция для применения фильтра
def apply_iir_filter(x, b, a):
    """Применяет IIR фильтр к сигналу x"""
    return signal.lfilter(b, a, x)

# Функция для подбора оптимальных параметров фильтра
def optimize_filter_parameters(d_freq):
    """Подбирает оптимальные параметры фильтра для заданной частоты помехи"""
    
    # Эмпирические правила для подбора параметров
    # T1 должен быть связан с частотой помехи
    T1 = 1.0 / (2 * np.pi * d_freq)  # базовая постоянная времени
    
    # T2 и T3 должны обеспечивать подавление помехи
    T2 = T1 * 0.5  # более быстрая постоянная времени
    T3 = T1 * 2.0  # более медленная постоянная времени
    
    return T1, T2, T3

# График 1: Сравнение сигналов во временной области
plt.figure(figsize=(15, 10))

for i, d in enumerate(d_values):
    # Создаем зашумлённый сигнал с гармонической помехой
    harmonic_noise = c * np.sin(2 * np.pi * d * t)
    u = g + harmonic_noise
    
    # Подбираем параметры фильтра
    T1, T2, T3 = optimize_filter_parameters(d)
    
    # Создаем фильтр
    b, a = create_special_filter(T1, T2, T3)
    
    # Применяем фильтр
    u_filtered = apply_iir_filter(u, b, a)
    
    plt.subplot(2, 2, i+1)
    mask_plot = (t >= t1-1) & (t <= t2+1)
    
    plt.plot(t[mask_plot], g[mask_plot], 'b-', linewidth=2, label='Исходный сигнал g(t)')
    plt.plot(t[mask_plot], u[mask_plot], 'r-', alpha=0.7, linewidth=1, label=f'С помехой (d={d} Гц)')
    plt.plot(t[mask_plot], u_filtered[mask_plot], 'g-', linewidth=2, label='Отфильтрованный')
    
    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.title(f'Специальный фильтр, d = {d} Гц')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task2/special_filter_time_domain.png', dpi=300, bbox_inches='tight')
plt.close()

# График 2: Фурье-образы
plt.figure(figsize=(15, 10))

for i, d in enumerate(d_values):
    harmonic_noise = c * np.sin(2 * np.pi * d * t)
    u = g + harmonic_noise
    
    T1, T2, T3 = optimize_filter_parameters(d)
    b, a = create_special_filter(T1, T2, T3)
    u_filtered = apply_iir_filter(u, b, a)
    
    # Вычисляем Фурье-образы
    U = fftshift(fft(u))
    U_filtered = fftshift(fft(u_filtered))
    
    # Создаем массив частот
    T_total = t[-1] - t[0]
    df = 1 / T_total
    f = np.linspace(-N//2, N//2, N, endpoint=False) * df
    
    plt.subplot(2, 2, i+1)
    
    # Показываем частоты до 30 Гц
    mask_freq = (f >= 0) & (f <= 30)
    
    plt.plot(f[mask_freq], np.abs(U[mask_freq]), 'r-', alpha=0.7, linewidth=1, label='Исходный сигнал')
    plt.plot(f[mask_freq], np.abs(U_filtered[mask_freq]), 'g-', linewidth=2, label='Отфильтрованный')
    plt.axvline(x=d, color='orange', linestyle='--', alpha=0.7, label=f'Частота помехи {d} Гц')
    
    plt.xlabel('Частота f')
    plt.ylabel('|U(f)|')
    plt.title(f'Фурье-образы, d = {d} Гц')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task2/special_filter_freq_domain.png', dpi=300, bbox_inches='tight')
plt.close()

# График 3: АЧХ фильтров
plt.figure(figsize=(15, 10))

for i, d in enumerate(d_values):
    T1, T2, T3 = optimize_filter_parameters(d)
    
    # Теоретическая АЧХ фильтра
    f_theoretical = np.logspace(-1, 2, 1000)
    omega = 2 * np.pi * f_theoretical
    
    # АЧХ: |W(jω)| = |(jωT1 + 1)^2 / ((jωT2 + 1)(jωT3 + 1))|
    numerator = np.sqrt((1 - (omega * T1)**2)**2 + (2 * omega * T1)**2)
    denominator = np.sqrt((1 - (omega * T2)**2) * (1 - (omega * T3)**2) + (omega * (T2 + T3))**2)
    
    magnitude = numerator / denominator
    
    plt.subplot(2, 2, i+1)
    plt.semilogx(f_theoretical, 20 * np.log10(magnitude), 'b-', linewidth=2, label=f'АЧХ фильтра')
    plt.axvline(x=d, color='red', linestyle='--', alpha=0.7, label=f'Частота помехи {d} Гц')
    
    plt.xlabel('Частота f (Гц)')
    plt.ylabel('АЧХ (дБ)')
    plt.title(f'АЧХ специального фильтра, d = {d} Гц')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-40, 10)

plt.tight_layout()
plt.savefig('../images/task2/special_filter_frequency_response.png', dpi=300, bbox_inches='tight')
plt.close()

# Анализ влияния параметра c на эффективность фильтрации
plt.figure(figsize=(15, 10))

d_fixed = 10  # фиксированная частота помехи
c_values = [0.2, 0.5, 0.8, 1.2]

for i, c_val in enumerate(c_values):
    harmonic_noise = c_val * np.sin(2 * np.pi * d_fixed * t)
    u_c = g + harmonic_noise
    
    T1, T2, T3 = optimize_filter_parameters(d_fixed)
    b, a = create_special_filter(T1, T2, T3)
    u_filtered_c = apply_iir_filter(u_c, b, a)
    
    plt.subplot(2, 2, i+1)
    mask_plot = (t >= t1-1) & (t <= t2+1)
    
    plt.plot(t[mask_plot], g[mask_plot], 'b-', linewidth=2, label='Исходный сигнал')
    plt.plot(t[mask_plot], u_c[mask_plot], 'r-', alpha=0.7, linewidth=1, label=f'С помехой (c={c_val})')
    plt.plot(t[mask_plot], u_filtered_c[mask_plot], 'g-', linewidth=2, label='Отфильтрованный')
    
    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.title(f'Влияние параметра c, c = {c_val}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task2/special_filter_c_influence.png', dpi=300, bbox_inches='tight')
plt.close()

# Количественный анализ эффективности фильтрации
print("Анализ специального фильтра:")
print("=" * 50)

for d in d_values:
    harmonic_noise = c * np.sin(2 * np.pi * d * t)
    u = g + harmonic_noise
    
    T1, T2, T3 = optimize_filter_parameters(d)
    b, a = create_special_filter(T1, T2, T3)
    u_filtered = apply_iir_filter(u, b, a)
    
    # Вычисляем среднеквадратичную ошибку
    mse = np.mean((g - u_filtered)**2)
    # Вычисляем корреляцию с исходным сигналом
    correlation = np.corrcoef(g, u_filtered)[0, 1]
    
    # Вычисляем подавление помехи
    noise_power_original = np.mean(harmonic_noise**2)
    noise_power_filtered = np.mean((u_filtered - g)**2)
    suppression = 10 * np.log10(noise_power_original / noise_power_filtered) if noise_power_filtered > 0 else float('inf')
    
    print(f"d = {d} Гц:")
    print(f"  Параметры фильтра: T1={T1:.3f}, T2={T2:.3f}, T3={T3:.3f}")
    print(f"  Среднеквадратичная ошибка: {mse:.6f}")
    print(f"  Корреляция с исходным сигналом: {correlation:.6f}")
    print(f"  Подавление помехи: {suppression:.2f} дБ")
    print()

print(f"Параметры: a={a}, t1={t1}, t2={t2}, b={b}, c={c}")
print(f"Исследованные значения d: {d_values}") 