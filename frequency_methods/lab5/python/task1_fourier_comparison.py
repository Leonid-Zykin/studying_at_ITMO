import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.integrate import trapz
import time
import os

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task1', exist_ok=True)

# Определяем прямоугольную функцию
def rect_function(t):
    """Прямоугольная функция Π(t)"""
    return np.where(np.abs(t) <= 0.5, 1, 0)

# Аналитический Фурье-образ (sinc функция)
def analytical_fourier(nu):
    """Аналитический Фурье-образ прямоугольной функции"""
    return np.sinc(nu)

# Функция для вычисления Фурье-образа численным интегрированием
def numerical_fourier_transform(t, signal, nu_array):
    """Вычисляет Фурье-образ с помощью численного интегрирования"""
    fourier_transform = []
    for nu in nu_array:
        integrand = signal * np.exp(-2j * np.pi * nu * t)
        result = trapz(integrand, t)
        fourier_transform.append(result)
    return np.array(fourier_transform)

# Функция для обратного Фурье-преобразования
def numerical_inverse_fourier(nu, fourier_signal, t_array):
    """Вычисляет обратное Фурье-преобразование с помощью численного интегрирования"""
    inverse_transform = []
    for t in t_array:
        integrand = fourier_signal * np.exp(2j * np.pi * nu * t)
        result = trapz(integrand, nu)
        inverse_transform.append(result)
    return np.array(inverse_transform)

# График 1: Сравнение аналитического и численного методов с комплексными образами
print("Создание графика сравнения методов с комплексными образами...")
plt.figure(figsize=(20, 15))

# Временная область
t_analytical = np.linspace(-2, 2, 1000)
nu_analytical = np.linspace(-10, 10, 1000)

# Исходная функция
plt.subplot(3, 3, 1)
plt.plot(t_analytical, rect_function(t_analytical), 'b-', linewidth=3, label='Π(t)')
plt.xlabel('Время t', fontsize=12)
plt.ylabel('Амплитуда', fontsize=12)
plt.title('Прямоугольная функция Π(t)', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Аналитический Фурье-образ (действительная часть)
analytical_fourier_vals = analytical_fourier(nu_analytical)
plt.subplot(3, 3, 2)
plt.plot(nu_analytical, np.real(analytical_fourier_vals), 'r-', linewidth=3, label='Re[sinc(ν)]')
plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Действительная часть', fontsize=12)
plt.title('Аналитический Фурье-образ (Re)', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)

# Аналитический Фурье-образ (мнимая часть)
plt.subplot(3, 3, 3)
plt.plot(nu_analytical, np.imag(analytical_fourier_vals), 'r-', linewidth=3, label='Im[sinc(ν)]')
plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Мнимая часть', fontsize=12)
plt.title('Аналитический Фурье-образ (Im)', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)

# Численное интегрирование
print("Вычисление Фурье-образа методом численного интегрирования...")
start_time = time.time()

# Параметры для численного интегрирования
t_trapz = np.arange(-10, 10, 0.01)
nu_trapz = np.arange(-10, 10, 0.05)

# Вычисление Фурье-образа с помощью trapz
fourier_trapz = numerical_fourier_transform(t_trapz, rect_function(t_trapz), nu_trapz)
trapz_time = time.time() - start_time

# Обратное преобразование с помощью trapz
t_reconstruct = np.arange(-3, 3, 0.05)
reconstructed_trapz = numerical_inverse_fourier(nu_trapz, fourier_trapz, t_reconstruct)

# Численное интегрирование (действительная часть)
plt.subplot(3, 3, 4)
plt.plot(nu_trapz, np.real(fourier_trapz), 'g-', linewidth=2, label='Re[Π̂(ν)] численное')
plt.plot(nu_analytical, np.real(analytical_fourier_vals), 'r--', linewidth=2, alpha=0.7, label='Re[sinc(ν)] аналитич.')
plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Действительная часть', fontsize=12)
plt.title('Численное интегрирование (Re)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)

# Численное интегрирование (мнимая часть)
plt.subplot(3, 3, 5)
plt.plot(nu_trapz, np.imag(fourier_trapz), 'g-', linewidth=2, label='Im[Π̂(ν)] численное')
plt.plot(nu_analytical, np.imag(analytical_fourier_vals), 'r--', linewidth=2, alpha=0.7, label='Im[sinc(ν)] аналитич.')
plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Мнимая часть', fontsize=12)
plt.title('Численное интегрирование (Im)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)

# Восстановленная функция
plt.subplot(3, 3, 6)
plt.plot(t_reconstruct, np.real(reconstructed_trapz), 'g-', linewidth=2, label='Восстановленная')
plt.plot(t_analytical, rect_function(t_analytical), 'r--', linewidth=2, alpha=0.7, label='Исходная')
plt.xlabel('Время t', fontsize=12)
plt.ylabel('Амплитуда', fontsize=12)
plt.title('Восстановление (численное)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-2, 2)

# DFT метод
print("Вычисление Фурье-образа методом DFT...")
start_time = time.time()

# Параметры для DFT
N = 1024
T = 20.0  # Общее время
dt = T / N
t_dft = np.linspace(-T/2, T/2, N)
signal_dft = rect_function(t_dft)

# Вычисление DFT
fourier_dft = fftshift(fft(signal_dft)) * dt
dft_time = time.time() - start_time

# Частотная ось для DFT
df = 1.0 / T
nu_dft = np.linspace(-1/(2*dt), 1/(2*dt), N)
nu_dft = fftshift(nu_dft)

# Обратное DFT
reconstructed_dft = ifft(ifftshift(fourier_dft / dt))

# DFT (действительная часть)
plt.subplot(3, 3, 7)
plt.plot(nu_dft, np.real(fourier_dft), 'b-', linewidth=2, label='Re[Π̂(ν)] DFT')
plt.plot(nu_analytical, np.real(analytical_fourier_vals), 'r--', linewidth=2, alpha=0.7, label='Re[sinc(ν)] аналитич.')
plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Действительная часть', fontsize=12)
plt.title('DFT метод (Re)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)

# DFT (мнимая часть)
plt.subplot(3, 3, 8)
plt.plot(nu_dft, np.imag(fourier_dft), 'b-', linewidth=2, label='Im[Π̂(ν)] DFT')
plt.plot(nu_analytical, np.imag(analytical_fourier_vals), 'r--', linewidth=2, alpha=0.7, label='Im[sinc(ν)] аналитич.')
plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Мнимая часть', fontsize=12)
plt.title('DFT метод (Im)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)

# Восстановленная функция DFT
plt.subplot(3, 3, 9)
plt.plot(t_dft, np.real(reconstructed_dft), 'b-', linewidth=2, label='Восстановленная DFT')
plt.plot(t_analytical, rect_function(t_analytical), 'r--', linewidth=2, alpha=0.7, label='Исходная')
plt.xlabel('Время t', fontsize=12)
plt.ylabel('Амплитуда', fontsize=12)
plt.title('Восстановление (DFT)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-2, 2)

plt.tight_layout()
plt.savefig('../images/task1/analytical_and_trapz_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# График 2: Исследование влияния параметров численного интегрирования
print("Создание графика исследования влияния параметров...")
plt.figure(figsize=(20, 12))

# Исследование влияния шага интегрирования
step_sizes = [0.1, 0.05, 0.01, 0.005]
colors = ['red', 'green', 'blue', 'orange']

plt.subplot(2, 3, 1)
for i, step in enumerate(step_sizes):
    t_step = np.arange(-10, 10, step)
    nu_step = np.arange(-5, 5, 0.1)
    fourier_step = numerical_fourier_transform(t_step, rect_function(t_step), nu_step)
    
    error = np.abs(fourier_step - analytical_fourier(nu_step))
    plt.plot(nu_step, error, color=colors[i], linewidth=2, label=f'dt = {step}')

plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Абсолютная ошибка', fontsize=12)
plt.title('Влияние шага интегрирования', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Исследование влияния размера промежутка
intervals = [5, 10, 20, 50]

plt.subplot(2, 3, 2)
for i, interval in enumerate(intervals):
    t_int = np.arange(-interval, interval, 0.02)
    nu_int = np.arange(-5, 5, 0.1)
    fourier_int = numerical_fourier_transform(t_int, rect_function(t_int), nu_int)
    
    error = np.abs(fourier_int - analytical_fourier(nu_int))
    plt.plot(nu_int, error, color=colors[i], linewidth=2, label=f'T = ±{interval}')

plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Абсолютная ошибка', fontsize=12)
plt.title('Влияние размера промежутка', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Зависимость времени выполнения от параметров
plt.subplot(2, 3, 3)
times = []
for step in step_sizes:
    t_time = np.arange(-10, 10, step)
    nu_time = np.arange(-5, 5, 0.1)
    
    start = time.time()
    _ = numerical_fourier_transform(t_time, rect_function(t_time), nu_time)
    exec_time = time.time() - start
    times.append(exec_time)

plt.bar(range(len(step_sizes)), times, color=colors, alpha=0.7)
plt.xticks(range(len(step_sizes)), [f'dt={s}' for s in step_sizes])
plt.xlabel('Шаг интегрирования', fontsize=12)
plt.ylabel('Время выполнения (с)', fontsize=12)
plt.title('Быстродействие vs точность', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Сравнение действительных частей для разных методов
plt.subplot(2, 3, 4)
plt.plot(nu_analytical, np.real(analytical_fourier_vals), 'r-', linewidth=3, label='Аналитический', alpha=0.8)
plt.plot(nu_trapz, np.real(fourier_trapz), 'g--', linewidth=2, label='Численное интегр.', alpha=0.8)
plt.plot(nu_dft, np.real(fourier_dft), 'b:', linewidth=2, label='DFT', alpha=0.8)
plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Re[Π̂(ν)]', fontsize=12)
plt.title('Сравнение действительных частей', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-3, 3)

# Сравнение мнимых частей для разных методов
plt.subplot(2, 3, 5)
plt.plot(nu_analytical, np.imag(analytical_fourier_vals), 'r-', linewidth=3, label='Аналитический', alpha=0.8)
plt.plot(nu_trapz, np.imag(fourier_trapz), 'g--', linewidth=2, label='Численное интегр.', alpha=0.8)
plt.plot(nu_dft, np.imag(fourier_dft), 'b:', linewidth=2, label='DFT', alpha=0.8)
plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Im[Π̂(ν)]', fontsize=12)
plt.title('Сравнение мнимых частей', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-3, 3)

# Анализ ошибок
plt.subplot(2, 3, 6)
# Интерполируем для сравнения на одной сетке
from scipy.interpolate import interp1d

nu_common = np.linspace(-3, 3, 500)
analytical_interp = analytical_fourier(nu_common)

# Интерполируем численные результаты
trapz_interp_func = interp1d(nu_trapz, fourier_trapz, kind='linear', bounds_error=False, fill_value=0)
trapz_interp = trapz_interp_func(nu_common)

dft_interp_func = interp1d(nu_dft, fourier_dft, kind='linear', bounds_error=False, fill_value=0)
dft_interp = dft_interp_func(nu_common)

error_trapz = np.abs(trapz_interp - analytical_interp)
error_dft = np.abs(dft_interp - analytical_interp)

plt.plot(nu_common, error_trapz, 'g-', linewidth=2, label=f'Численное (макс: {np.max(error_trapz):.4f})')
plt.plot(nu_common, error_dft, 'b-', linewidth=2, label=f'DFT (макс: {np.max(error_dft):.4f})')
plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Абсолютная ошибка', fontsize=12)
plt.title('Сравнение ошибок методов', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.savefig('../images/task1/detailed_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# График 3: Приближение непрерывного с помощью DFT (умное использование)
print("Создание графика точного непрерывного преобразования с помощью DFT...")
plt.figure(figsize=(20, 15))

# Правильное масштабирование для получения непрерывного преобразования
# Ключевые принципы:
# 1. dt * df = 1/N (связь между временным и частотным разрешением)
# 2. Для получения непрерывного преобразования нужно умножить на dt
# 3. Для обратного преобразования нужно разделить на dt

# Параметры для точного преобразования
T_precise = 40  # увеличиваем интервал для лучшей точности
N_precise = 4096  # увеличиваем количество точек
dt_precise = T_precise / N_precise
df_precise = 1.0 / T_precise

print(f"Параметры точного DFT:")
print(f"  T = {T_precise}, N = {N_precise}")
print(f"  dt = {dt_precise:.6f}, df = {df_precise:.6f}")
print(f"  dt * df * N = {dt_precise * df_precise * N_precise}")

t_precise = np.linspace(-T_precise/2, T_precise/2, N_precise, endpoint=False)
signal_precise = rect_function(t_precise)

# Прямое DFT преобразование с правильным масштабированием
fourier_precise_raw = fft(signal_precise)
fourier_precise_shifted = fftshift(fourier_precise_raw)
fourier_precise_scaled = fourier_precise_shifted * dt_precise  # Масштабирование для непрерывного преобразования

# Создаем правильный массив частот
nu_precise = fftshift(np.fft.fftfreq(N_precise, dt_precise))

# Обратное преобразование
reconstructed_precise_raw = ifft(ifftshift(fourier_precise_scaled / dt_precise))
reconstructed_precise = np.real(reconstructed_precise_raw)

# Вычисляем ошибки
analytical_precise = analytical_fourier(nu_precise)
error_fourier = np.abs(fourier_precise_scaled - analytical_precise)
error_time = np.abs(reconstructed_precise - signal_precise)

print(f"Ошибки точного DFT:")
print(f"  Максимальная ошибка в частотной области: {np.max(error_fourier):.6f}")
print(f"  Средняя ошибка в частотной области: {np.mean(error_fourier):.6f}")
print(f"  Максимальная ошибка во временной области: {np.max(error_time):.6f}")
print(f"  Средняя ошибка во временной области: {np.mean(error_time):.6f}")

# Подграфик 1: Исходная функция
plt.subplot(3, 3, 1)
plt.plot(t_precise, signal_precise, 'b-', linewidth=2, label='Π(t)')
plt.xlabel('Время t', fontsize=12)
plt.ylabel('Амплитуда', fontsize=12)
plt.title('Исходная функция', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-3, 3)

# Подграфик 2: Сравнение действительных частей
plt.subplot(3, 3, 2)
plt.plot(nu_precise, np.real(fourier_precise_scaled), 'b-', linewidth=2, label='Re[Π̂(ν)] DFT точное')
plt.plot(nu_analytical, np.real(analytical_fourier_vals), 'r--', linewidth=2, alpha=0.8, label='Re[sinc(ν)] аналитич.')
plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Действительная часть', fontsize=12)
plt.title('Точное DFT (Re)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)

# Подграфик 3: Сравнение мнимых частей
plt.subplot(3, 3, 3)
plt.plot(nu_precise, np.imag(fourier_precise_scaled), 'b-', linewidth=2, label='Im[Π̂(ν)] DFT точное')
plt.plot(nu_analytical, np.imag(analytical_fourier_vals), 'r--', linewidth=2, alpha=0.8, label='Im[sinc(ν)] аналитич.')
plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Мнимая часть', fontsize=12)
plt.title('Точное DFT (Im)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)

# Подграфик 4: Восстановленная функция
plt.subplot(3, 3, 4)
plt.plot(t_precise, reconstructed_precise, 'b-', linewidth=2, label='Восстановленная')
plt.plot(t_precise, signal_precise, 'r--', linewidth=2, alpha=0.8, label='Исходная')
plt.xlabel('Время t', fontsize=12)
plt.ylabel('Амплитуда', fontsize=12)
plt.title('Восстановление (точное DFT)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-3, 3)

# Подграфик 5: Ошибка в частотной области
plt.subplot(3, 3, 5)
plt.semilogy(nu_precise, error_fourier, 'g-', linewidth=2, label=f'Макс: {np.max(error_fourier):.2e}')
plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Абсолютная ошибка', fontsize=12)
plt.title('Ошибка в частотной области', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)

# Подграфик 6: Ошибка во временной области
plt.subplot(3, 3, 6)
plt.semilogy(t_precise, error_time, 'g-', linewidth=2, label=f'Макс: {np.max(error_time):.2e}')
plt.xlabel('Время t', fontsize=12)
plt.ylabel('Абсолютная ошибка', fontsize=12)
plt.title('Ошибка во временной области', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-3, 3)

# Подграфик 7: Сравнение всех методов (модуль)
plt.subplot(3, 3, 7)
plt.plot(nu_analytical, np.abs(analytical_fourier_vals), 'r-', linewidth=3, label='Аналитический', alpha=0.8)
plt.plot(nu_trapz, np.abs(fourier_trapz), 'g--', linewidth=2, label='Численное интегр.', alpha=0.8)
plt.plot(nu_dft, np.abs(fourier_dft), 'b:', linewidth=2, label='DFT простое', alpha=0.8)
plt.plot(nu_precise, np.abs(fourier_precise_scaled), 'purple', linestyle='-', linewidth=2, label='DFT точное', alpha=0.8)
plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('|Π̂(ν)|', fontsize=12)
plt.title('Сравнение модулей', fontsize=14, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim(-3, 3)

# Подграфик 8: Быстродействие
start_precise = time.time()
_ = fftshift(fft(signal_precise)) * dt_precise
precise_time = time.time() - start_precise

plt.subplot(3, 3, 8)
methods = ['Численное\nинтегрирование', 'DFT\nпростое', 'DFT\nточное']
times = [trapz_time, dft_time, precise_time]
colors = ['green', 'blue', 'purple']
bars = plt.bar(methods, times, color=colors, alpha=0.7)

# Добавляем значения на столбцы
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
             f'{time_val:.4f}с', ha='center', va='bottom', fontsize=10)

plt.ylabel('Время выполнения (с)', fontsize=12)
plt.title('Сравнение быстродействия', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Подграфик 9: Точность
plt.subplot(3, 3, 9)
# Вычисляем ошибки для всех методов на общей сетке
nu_common_fine = np.linspace(-3, 3, 1000)
analytical_common = analytical_fourier(nu_common_fine)

# Интерполируем результаты всех методов
from scipy.interpolate import interp1d

trapz_interp = interp1d(nu_trapz, fourier_trapz, kind='linear', bounds_error=False, fill_value=0)(nu_common_fine)
dft_interp = interp1d(nu_dft, fourier_dft, kind='linear', bounds_error=False, fill_value=0)(nu_common_fine)
precise_interp = interp1d(nu_precise, fourier_precise_scaled, kind='linear', bounds_error=False, fill_value=0)(nu_common_fine)

error_trapz_fine = np.abs(trapz_interp - analytical_common)
error_dft_fine = np.abs(dft_interp - analytical_common)
error_precise_fine = np.abs(precise_interp - analytical_common)

methods = ['Численное\nинтегрирование', 'DFT\nпростое', 'DFT\nточное']
max_errors = [np.max(error_trapz_fine), np.max(error_dft_fine), np.max(error_precise_fine)]
mean_errors = [np.mean(error_trapz_fine), np.mean(error_dft_fine), np.mean(error_precise_fine)]

x = np.arange(len(methods))
width = 0.35

bars1 = plt.bar(x - width/2, max_errors, width, label='Максимальная ошибка', color='red', alpha=0.7)
bars2 = plt.bar(x + width/2, mean_errors, width, label='Средняя ошибка', color='blue', alpha=0.7)

plt.xlabel('Метод', fontsize=12)
plt.ylabel('Абсолютная ошибка', fontsize=12)
plt.title('Сравнение точности', fontsize=14, fontweight='bold')
plt.xticks(x, methods)
plt.legend(fontsize=10)
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task1/precise_fourier_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# График 4: Объяснение принципов масштабирования DFT
print("Создание графика объяснения принципов DFT...")
plt.figure(figsize=(18, 12))

# Демонстрация влияния параметров на точность DFT
T_values = [10, 20, 40, 80]
N_values = [512, 1024, 2048, 4096]

plt.subplot(2, 3, 1)
for i, T_val in enumerate(T_values):
    N_demo = 1024
    dt_demo = T_val / N_demo
    t_demo = np.linspace(-T_val/2, T_val/2, N_demo, endpoint=False)
    signal_demo = rect_function(t_demo)
    
    fourier_demo = fftshift(fft(signal_demo)) * dt_demo
    nu_demo = fftshift(np.fft.fftfreq(N_demo, dt_demo))
    
    error_demo = np.abs(fourier_demo - analytical_fourier(nu_demo))
    plt.semilogy(nu_demo, error_demo, linewidth=2, label=f'T = {T_val}')

plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Абсолютная ошибка', fontsize=12)
plt.title('Влияние интервала T на точность', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-3, 3)

plt.subplot(2, 3, 2)
for i, N_val in enumerate(N_values):
    T_demo = 40
    dt_demo = T_demo / N_val
    t_demo = np.linspace(-T_demo/2, T_demo/2, N_val, endpoint=False)
    signal_demo = rect_function(t_demo)
    
    fourier_demo = fftshift(fft(signal_demo)) * dt_demo
    nu_demo = fftshift(np.fft.fftfreq(N_val, dt_demo))
    
    error_demo = np.abs(fourier_demo - analytical_fourier(nu_demo))
    plt.semilogy(nu_demo, error_demo, linewidth=2, label=f'N = {N_val}')

plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('Абсолютная ошибка', fontsize=12)
plt.title('Влияние количества точек N на точность', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-3, 3)

# Демонстрация правильного масштабирования
plt.subplot(2, 3, 3)
N_scale = 1024
T_scale = 20
dt_scale = T_scale / N_scale
t_scale = np.linspace(-T_scale/2, T_scale/2, N_scale, endpoint=False)
signal_scale = rect_function(t_scale)

# Различные варианты масштабирования
fourier_raw = fftshift(fft(signal_scale))  # Без масштабирования
fourier_dt = fftshift(fft(signal_scale)) * dt_scale  # С dt
fourier_df = fftshift(fft(signal_scale)) / N_scale  # С df
fourier_sqrt = fftshift(fft(signal_scale)) / np.sqrt(N_scale)  # Унитарное

nu_scale = fftshift(np.fft.fftfreq(N_scale, dt_scale))
analytical_scale = analytical_fourier(nu_scale)

plt.plot(nu_scale, np.abs(fourier_raw), 'r-', linewidth=2, label='Без масштабирования', alpha=0.7)
plt.plot(nu_scale, np.abs(fourier_dt), 'b-', linewidth=2, label='× dt (правильно)', alpha=0.7)
plt.plot(nu_scale, np.abs(fourier_df), 'g-', linewidth=2, label='÷ N', alpha=0.7)
plt.plot(nu_scale, np.abs(fourier_sqrt), 'orange', linewidth=2, label='÷ √N (унитарное)', alpha=0.7)
plt.plot(nu_scale, np.abs(analytical_scale), 'k--', linewidth=3, label='Аналитический', alpha=0.8)

plt.xlabel('Частота ν', fontsize=12)
plt.ylabel('|Π̂(ν)|', fontsize=12)
plt.title('Варианты масштабирования DFT', fontsize=14, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim(-3, 3)

# Демонстрация обратного преобразования
plt.subplot(2, 3, 4)
# Правильное обратное преобразование
reconstructed_correct = np.real(ifft(ifftshift(fourier_dt / dt_scale)))
reconstructed_wrong1 = np.real(ifft(ifftshift(fourier_dt)))
reconstructed_wrong2 = np.real(ifft(ifftshift(fourier_raw / N_scale)))

plt.plot(t_scale, signal_scale, 'k-', linewidth=3, label='Исходная', alpha=0.8)
plt.plot(t_scale, reconstructed_correct, 'b-', linewidth=2, label='Правильное обратное', alpha=0.7)
plt.plot(t_scale, reconstructed_wrong1, 'r--', linewidth=2, label='Неправильное 1', alpha=0.7)
plt.plot(t_scale, reconstructed_wrong2, 'g:', linewidth=2, label='Неправильное 2', alpha=0.7)

plt.xlabel('Время t', fontsize=12)
plt.ylabel('Амплитуда', fontsize=12)
plt.title('Обратное преобразование', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-3, 3)

# Схема алгоритма
plt.subplot(2, 3, 5)
plt.text(0.1, 0.9, 'АЛГОРИТМ ТОЧНОГО DFT:', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.1, 0.8, '1. Выбираем T и N:', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.15, 0.75, '   dt = T/N, df = 1/T', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.65, '2. Прямое преобразование:', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.15, 0.6, '   F̂(ν) = fftshift(fft(f(t))) × dt', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.5, '3. Обратное преобразование:', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.15, 0.45, '   f(t) = ifft(ifftshift(F̂(ν)/dt))', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.35, '4. Частотная ось:', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.15, 0.3, '   ν = fftshift(fftfreq(N, dt))', fontsize=11, transform=plt.gca().transAxes)
plt.text(0.1, 0.2, 'КЛЮЧ: масштабирование на dt!', fontsize=12, fontweight='bold', color='red', transform=plt.gca().transAxes)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

# Сравнительная таблица результатов
plt.subplot(2, 3, 6)
methods_table = ['Аналитический', 'Численное\nинтегрирование', 'DFT простое', 'DFT точное']
times_table = [0, trapz_time, dft_time, precise_time]
errors_table = [0, np.mean(error_trapz_fine), np.mean(error_dft_fine), np.mean(error_precise_fine)]

# Создаем таблицу
table_data = []
for i, method in enumerate(methods_table):
    table_data.append([method, f'{times_table[i]:.4f}', f'{errors_table[i]:.2e}'])

table = plt.table(cellText=table_data,
                  colLabels=['Метод', 'Время (с)', 'Средняя ошибка'],
                  cellLoc='center',
                  loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Раскрашиваем лучшие результаты
table[(1, 1)].set_facecolor('#ffcccc')  # Худшее время
table[(4, 1)].set_facecolor('#ccffcc')  # Лучшее время
table[(1, 2)].set_facecolor('#ccffcc')  # Лучшая точность
table[(4, 2)].set_facecolor('#ccffcc')  # Лучшая точность

plt.title('Сводная таблица результатов', fontsize=14, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('../images/task1/dft_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Анализ задания 1 завершен!")
print("="*60)
print("ВЫВОДЫ:")
print(f"1. Численное интегрирование: время {trapz_time:.4f}с, точное, но медленное")
print(f"2. DFT простое: время {dft_time:.6f}с, быстрое, но требует правильного масштабирования")
print(f"3. DFT точное: время {precise_time:.6f}с, быстрое и точное при правильном масштабировании")
print(f"4. Ускорение DFT vs численное интегрирование: {trapz_time/precise_time:.0f}x")
print("="*60) 