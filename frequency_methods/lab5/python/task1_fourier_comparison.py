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

# График 1: Истинная функция и её Фурье-образ
plt.figure(figsize=(15, 10))

# Временная область
t_analytical = np.linspace(-2, 2, 1000)
nu_analytical = np.linspace(-10, 10, 1000)

plt.subplot(2, 2, 1)
plt.plot(t_analytical, rect_function(t_analytical), 'b-', linewidth=2, label='Π(t)')
plt.xlabel('Время t')
plt.ylabel('Амплитуда')
plt.title('Прямоугольная функция Π(t)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(nu_analytical, analytical_fourier(nu_analytical), 'r-', linewidth=2, label='sinc(ν)')
plt.xlabel('Частота ν')
plt.ylabel('Амплитуда')
plt.title('Аналитический Фурье-образ sinc(ν)')
plt.legend()
plt.grid(True, alpha=0.3)

# Численное интегрирование
print("Вычисление Фурье-образа методом численного интегрирования...")
start_time = time.time()

# Параметры для численного интегрирования
t_trapz = np.arange(-10, 10, 0.01)
nu_trapz = np.arange(-20, 20, 0.01)

# Вычисление Фурье-образа с помощью trapz
fourier_trapz = []
for nu in nu_trapz:
    integrand = rect_function(t_trapz) * np.exp(-2j * np.pi * nu * t_trapz)
    result = trapz(integrand, t_trapz)
    fourier_trapz.append(result)

fourier_trapz = np.array(fourier_trapz)
trapz_time = time.time() - start_time

# Обратное преобразование с помощью trapz
reconstructed_trapz = []
for t in t_trapz:
    integrand = fourier_trapz * np.exp(2j * np.pi * nu_trapz * t)
    result = trapz(integrand, nu_trapz)
    reconstructed_trapz.append(result)

reconstructed_trapz = np.array(reconstructed_trapz)

plt.subplot(2, 2, 3)
plt.plot(nu_trapz, np.abs(fourier_trapz), 'g-', linewidth=2, label='Численное интегрирование')
plt.plot(nu_analytical, np.abs(analytical_fourier(nu_analytical)), 'r--', linewidth=2, label='Аналитический')
plt.xlabel('Частота ν')
plt.ylabel('|Π̂(ν)|')
plt.title('Сравнение Фурье-образов (численное интегрирование)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)  # Ограничиваем диапазон для лучшей видимости

plt.subplot(2, 2, 4)
plt.plot(t_trapz, np.real(reconstructed_trapz), 'g-', linewidth=2, label='Восстановленная')
plt.plot(t_analytical, rect_function(t_analytical), 'r--', linewidth=2, label='Исходная')
plt.xlabel('Время t')
plt.ylabel('Амплитуда')
plt.title('Восстановление функции (численное интегрирование)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-2, 2)  # Ограничиваем диапазон для лучшей видимости

plt.tight_layout()
plt.savefig('../images/task1/analytical_and_trapz_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# DFT метод
print("Вычисление Фурье-образа методом DFT...")
start_time = time.time()

# Параметры для DFT
N = 2048
t_dft = np.linspace(-10, 10, N)
dt = t_dft[1] - t_dft[0]

# Создаем сигнал
signal_dft = rect_function(t_dft)

# Вычисляем Фурье-образ с помощью DFT
fourier_dft = fftshift(fft(signal_dft))

# Создаем массив частот
nu_dft = fftshift(np.fft.fftfreq(N, dt))

# Обратное преобразование
reconstructed_dft = ifft(ifftshift(fourier_dft))

dft_time = time.time() - start_time

# График 2: DFT метод
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(t_dft, signal_dft, 'b-', linewidth=2, label='Исходный сигнал')
plt.xlabel('Время t')
plt.ylabel('Амплитуда')
plt.title('Прямоугольная функция (дискретизированная)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(nu_dft, np.abs(fourier_dft), 'g-', linewidth=2, label='DFT')
plt.plot(nu_analytical, np.abs(analytical_fourier(nu_analytical)), 'r--', linewidth=2, label='Аналитический')
plt.xlabel('Частота ν')
plt.ylabel('|Π̂(ν)|')
plt.title('Сравнение Фурье-образов (DFT)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(t_dft, np.real(reconstructed_dft), 'g-', linewidth=2, label='Восстановленная')
plt.plot(t_analytical, rect_function(t_analytical), 'r--', linewidth=2, label='Исходная')
plt.xlabel('Время t')
plt.ylabel('Амплитуда')
plt.title('Восстановление функции (DFT)')
plt.legend()
plt.grid(True, alpha=0.3)

# Сравнение быстродействия
plt.subplot(2, 2, 4)
methods = ['Численное интегрирование', 'DFT']
times = [trapz_time, dft_time]
plt.bar(methods, times, color=['green', 'blue'])
plt.ylabel('Время выполнения (с)')
plt.title('Сравнение быстродействия методов')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task1/dft_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Умное использование DFT для получения точного непрерывного преобразования
print("Вычисление точного непрерывного преобразования с помощью DFT...")

# Правильное масштабирование для получения непрерывного преобразования
# Связь между дискретными и непрерывными переменными:
# dt * df = 1/N, где dt - шаг по времени, df - шаг по частоте, N - количество точек

# Параметры для точного преобразования
T = 20  # общий интервал времени
N_precise = 4096
dt_precise = T / N_precise
df_precise = 1 / T

t_precise = np.linspace(-T/2, T/2, N_precise, endpoint=False)
signal_precise = rect_function(t_precise)

# Вычисляем DFT
fourier_precise = fftshift(fft(signal_precise))

# Создаем правильный массив частот
nu_precise = fftshift(np.fft.fftfreq(N_precise, dt_precise))

# Масштабируем для получения непрерывного преобразования
fourier_precise_scaled = fourier_precise * dt_precise

# Обратное преобразование
reconstructed_precise = ifft(ifftshift(fourier_precise_scaled)) / dt_precise

# График 3: Точное преобразование
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(t_precise, signal_precise, 'b-', linewidth=2, label='Исходный сигнал')
plt.xlabel('Время t')
plt.ylabel('Амплитуда')
plt.title('Прямоугольная функция (точное преобразование)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(nu_precise, np.abs(fourier_precise_scaled), 'g-', linewidth=2, label='DFT (масштабированный)')
plt.plot(nu_analytical, np.abs(analytical_fourier(nu_analytical)), 'r--', linewidth=2, label='Аналитический')
plt.xlabel('Частота ν')
plt.ylabel('|Π̂(ν)|')
plt.title('Точное непрерывное преобразование')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(t_precise, np.real(reconstructed_precise), 'g-', linewidth=2, label='Восстановленная')
plt.plot(t_analytical, rect_function(t_analytical), 'r--', linewidth=2, label='Исходная')
plt.xlabel('Время t')
plt.ylabel('Амплитуда')
plt.title('Восстановление функции (точное преобразование)')
plt.legend()
plt.grid(True, alpha=0.3)

# Сравнение ошибок
plt.subplot(2, 2, 4)
methods = ['Численное интегрирование', 'DFT', 'Точное DFT']
error_trapz = np.mean(np.abs(np.real(reconstructed_trapz) - rect_function(t_trapz)))
error_dft = np.mean(np.abs(np.real(reconstructed_dft) - rect_function(t_dft)))
error_precise = np.mean(np.abs(np.real(reconstructed_precise) - rect_function(t_precise)))

errors = [error_trapz, error_dft, error_precise]
plt.bar(methods, errors, color=['green', 'blue', 'red'])
plt.ylabel('Средняя абсолютная ошибка')
plt.title('Сравнение точности методов')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task1/precise_fourier_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Детальный анализ ошибок
plt.figure(figsize=(15, 10))

# Ошибки в частотной области
plt.subplot(2, 2, 1)
mask_freq = (nu_trapz >= -5) & (nu_trapz <= 5)
error_freq_trapz = np.abs(fourier_trapz[mask_freq] - analytical_fourier(nu_trapz[mask_freq]))
plt.plot(nu_trapz[mask_freq], error_freq_trapz, 'g-', linewidth=2, label='Численное интегрирование')

mask_freq_dft = (nu_dft >= -5) & (nu_dft <= 5)
error_freq_dft = np.abs(fourier_dft[mask_freq_dft] - analytical_fourier(nu_dft[mask_freq_dft]))
plt.plot(nu_dft[mask_freq_dft], error_freq_dft, 'b-', linewidth=2, label='DFT')

mask_freq_precise = (nu_precise >= -5) & (nu_precise <= 5)
error_freq_precise = np.abs(fourier_precise_scaled[mask_freq_precise] - analytical_fourier(nu_precise[mask_freq_precise]))
plt.plot(nu_precise[mask_freq_precise], error_freq_precise, 'r-', linewidth=2, label='Точное DFT')

plt.xlabel('Частота ν')
plt.ylabel('Абсолютная ошибка')
plt.title('Ошибки в частотной области')
plt.legend()
plt.grid(True, alpha=0.3)

# Ошибки во временной области
plt.subplot(2, 2, 2)
mask_time = (t_trapz >= -2) & (t_trapz <= 2)
error_time_trapz = np.abs(np.real(reconstructed_trapz[mask_time]) - rect_function(t_trapz[mask_time]))
plt.plot(t_trapz[mask_time], error_time_trapz, 'g-', linewidth=2, label='Численное интегрирование')

mask_time_dft = (t_dft >= -2) & (t_dft <= 2)
error_time_dft = np.abs(np.real(reconstructed_dft[mask_time_dft]) - rect_function(t_dft[mask_time_dft]))
plt.plot(t_dft[mask_time_dft], error_time_dft, 'b-', linewidth=2, label='DFT')

mask_time_precise = (t_precise >= -2) & (t_precise <= 2)
error_time_precise = np.abs(np.real(reconstructed_precise[mask_time_precise]) - rect_function(t_precise[mask_time_precise]))
plt.plot(t_precise[mask_time_precise], error_time_precise, 'r-', linewidth=2, label='Точное DFT')

plt.xlabel('Время t')
plt.ylabel('Абсолютная ошибка')
plt.title('Ошибки во временной области')
plt.legend()
plt.grid(True, alpha=0.3)

# Сравнение статистик
plt.subplot(2, 2, 3)
methods = ['Численное интегрирование', 'DFT', 'Точное DFT']
mse_freq = [np.mean(error_freq_trapz**2), np.mean(error_freq_dft**2), np.mean(error_freq_precise**2)]
mse_time = [np.mean(error_time_trapz**2), np.mean(error_time_dft**2), np.mean(error_time_precise**2)]

x = np.arange(len(methods))
width = 0.35

plt.bar(x - width/2, mse_freq, width, label='Ошибка в частотной области', color='skyblue')
plt.bar(x + width/2, mse_time, width, label='Ошибка во временной области', color='lightcoral')

plt.xlabel('Метод')
plt.ylabel('Среднеквадратичная ошибка')
plt.title('Сравнение точности методов')
plt.xticks(x, methods, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Сравнение быстродействия и точности
plt.subplot(2, 2, 4)
# Добавляем время для точного DFT (аналогично обычному DFT)
times_all = [trapz_time, dft_time, dft_time]
plt.scatter(times_all, errors, s=100, c=['green', 'blue', 'red'], alpha=0.7)
for i, method in enumerate(methods):
    plt.annotate(method, (times_all[i], errors[i]), xytext=(5, 5), textcoords='offset points')

plt.xlabel('Время выполнения (с)')
plt.ylabel('Средняя абсолютная ошибка')
plt.title('Быстродействие vs Точность')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task1/detailed_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Вывод результатов
print("Анализ методов вычисления Фурье-образа:")
print("=" * 60)
print(f"Численное интегрирование:")
print(f"  Время выполнения: {trapz_time:.4f} с")
print(f"  Средняя ошибка во временной области: {error_trapz:.6f}")
print(f"  Средняя ошибка в частотной области: {np.mean(error_freq_trapz):.6f}")
print()
print(f"DFT:")
print(f"  Время выполнения: {dft_time:.4f} с")
print(f"  Средняя ошибка во временной области: {error_dft:.6f}")
print(f"  Средняя ошибка в частотной области: {np.mean(error_freq_dft):.6f}")
print()
print(f"Точное DFT:")
print(f"  Время выполнения: {dft_time:.4f} с (аналогично DFT)")
print(f"  Средняя ошибка во временной области: {error_precise:.6f}")
print(f"  Средняя ошибка в частотной области: {np.mean(error_freq_precise):.6f}")
print()
print("Выводы:")
print("- Численное интегрирование: медленно, но может быть точным при правильных параметрах")
print("- DFT: быстро, но требует правильного масштабирования для точности")
print("- Точное DFT: быстро и точно при правильном масштабировании") 