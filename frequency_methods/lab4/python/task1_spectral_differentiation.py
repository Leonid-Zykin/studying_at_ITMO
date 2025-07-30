import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.integrate import trapz
import os

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task1', exist_ok=True)

# Параметры эксперимента
dt = 0.1
t = np.arange(-100, 100 + dt, dt)
N = len(t)

# Исходный сигнал
y_clean = np.sin(t)

# Добавляем шум
np.random.seed(42)  # для воспроизводимости результатов
a = 0.1  # амплитуда шума
noise = a * (np.random.rand(N) - 0.5)
y_noisy = y_clean + noise

# 1. Численное дифференцирование
y_numerical_diff = np.zeros_like(y_noisy)
for i in range(N - 1):
    y_numerical_diff[i] = (y_noisy[i + 1] - y_noisy[i]) / dt
y_numerical_diff[-1] = y_numerical_diff[-2]  # для последней точки

# 2. Спектральное дифференцирование
# Создаем массив частот
T = t[-1] - t[0]  # общий интервал времени
df = 1 / T  # шаг по частоте
f = np.arange(-N//2, N//2) * df
if len(f) != N:
    f = np.linspace(-N//2, N//2, N, endpoint=False) * df

# Вычисляем Фурье-образ с помощью численного интегрирования
Y_fft = fftshift(fft(y_noisy))

# Умножаем на i*omega для получения спектральной производной
omega = 2 * np.pi * f
Y_diff_spectral = 1j * omega * Y_fft

# Обратное преобразование Фурье
y_spectral_diff = np.real(ifft(ifftshift(Y_diff_spectral)))

# 3. Истинная производная
y_true_diff = np.cos(t)

# График 1: Сравнение сигналов во временной области
plt.figure(figsize=(15, 10))

# Исходный сигнал
plt.subplot(2, 2, 1)
plt.plot(t, y_clean, 'b-', linewidth=2, label='Исходный сигнал sin(t)')
plt.plot(t, y_noisy, 'r-', alpha=0.7, linewidth=1, label='Зашумлённый сигнал')
plt.xlabel('Время t')
plt.ylabel('Амплитуда')
plt.title('Исходный и зашумлённый сигналы')
plt.legend()
plt.grid(True, alpha=0.3)

# Сравнение производных
plt.subplot(2, 2, 2)
plt.plot(t, y_true_diff, 'b-', linewidth=2, label='Истинная производная cos(t)')
plt.plot(t, y_numerical_diff, 'r-', alpha=0.7, linewidth=1, label='Численная производная')
plt.plot(t, y_spectral_diff, 'g-', alpha=0.7, linewidth=1, label='Спектральная производная')
plt.xlabel('Время t')
plt.ylabel('Амплитуда')
plt.title('Сравнение методов дифференцирования')
plt.legend()
plt.grid(True, alpha=0.3)

# Ошибки дифференцирования
plt.subplot(2, 2, 3)
error_numerical = np.abs(y_true_diff - y_numerical_diff)
error_spectral = np.abs(y_true_diff - y_spectral_diff)
plt.plot(t, error_numerical, 'r-', linewidth=1, label='Ошибка численного дифференцирования')
plt.plot(t, error_spectral, 'g-', linewidth=1, label='Ошибка спектрального дифференцирования')
plt.xlabel('Время t')
plt.ylabel('Абсолютная ошибка')
plt.title('Ошибки дифференцирования')
plt.legend()
plt.grid(True, alpha=0.3)

# Статистика ошибок
plt.subplot(2, 2, 4)
methods = ['Численное', 'Спектральное']
mse_values = [np.mean(error_numerical**2), np.mean(error_spectral**2)]
max_error_values = [np.max(error_numerical), np.max(error_spectral)]

x = np.arange(len(methods))
width = 0.35

plt.bar(x - width/2, mse_values, width, label='Среднеквадратичная ошибка', color='skyblue')
plt.bar(x + width/2, max_error_values, width, label='Максимальная ошибка', color='lightcoral')

plt.xlabel('Метод дифференцирования')
plt.ylabel('Ошибка')
plt.title('Статистика ошибок')
plt.xticks(x, methods)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task1/spectral_differentiation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# График 2: Фурье-образы
plt.figure(figsize=(15, 10))

# Фурье-образ исходного сигнала
plt.subplot(2, 2, 1)
mask_freq = (f >= 0) & (f <= 2)  # показываем только низкие частоты
plt.plot(f[mask_freq], np.abs(Y_fft[mask_freq]), 'b-', linewidth=2, label='Исходный сигнал')
plt.xlabel('Частота f')
plt.ylabel('|Y(f)|')
plt.title('Фурье-образ исходного сигнала')
plt.legend()
plt.grid(True, alpha=0.3)

# Фурье-образ спектральной производной
plt.subplot(2, 2, 2)
plt.plot(f[mask_freq], np.abs(Y_diff_spectral[mask_freq]), 'g-', linewidth=2, label='Спектральная производная')
plt.xlabel('Частота f')
plt.ylabel('|Y\'(f)|')
plt.title('Фурье-образ спектральной производной')
plt.legend()
plt.grid(True, alpha=0.3)

# Вещественная часть Фурье-образов
plt.subplot(2, 2, 3)
plt.plot(f[mask_freq], np.real(Y_fft[mask_freq]), 'b-', linewidth=2, label='Исходный сигнал')
plt.plot(f[mask_freq], np.real(Y_diff_spectral[mask_freq]), 'g-', linewidth=2, label='Спектральная производная')
plt.xlabel('Частота f')
plt.ylabel('Re(Y(f))')
plt.title('Вещественная часть Фурье-образов')
plt.legend()
plt.grid(True, alpha=0.3)

# Мнимая часть Фурье-образов
plt.subplot(2, 2, 4)
plt.plot(f[mask_freq], np.imag(Y_fft[mask_freq]), 'b-', linewidth=2, label='Исходный сигнал')
plt.plot(f[mask_freq], np.imag(Y_diff_spectral[mask_freq]), 'g-', linewidth=2, label='Спектральная производная')
plt.xlabel('Частота f')
plt.ylabel('Im(Y(f))')
plt.title('Мнимая часть Фурье-образов')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task1/spectral_differentiation_fourier.png', dpi=300, bbox_inches='tight')
plt.close()

# Детальный анализ в окрестности основной частоты
plt.figure(figsize=(15, 10))

# Основная частота сигнала sin(t) равна 1/(2π) ≈ 0.159 Гц
main_freq = 1 / (2 * np.pi)
freq_range = 0.1  # диапазон вокруг основной частоты

mask_main = (np.abs(f - main_freq) <= freq_range) | (np.abs(f + main_freq) <= freq_range)

plt.subplot(2, 2, 1)
plt.plot(f[mask_main], np.abs(Y_fft[mask_main]), 'b-', linewidth=2, label='Исходный сигнал')
plt.axvline(x=main_freq, color='red', linestyle='--', alpha=0.7, label=f'Основная частота {main_freq:.3f} Гц')
plt.axvline(x=-main_freq, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Частота f')
plt.ylabel('|Y(f)|')
plt.title('Фурье-образ в окрестности основной частоты')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(f[mask_main], np.abs(Y_diff_spectral[mask_main]), 'g-', linewidth=2, label='Спектральная производная')
plt.axvline(x=main_freq, color='red', linestyle='--', alpha=0.7, label=f'Основная частота {main_freq:.3f} Гц')
plt.axvline(x=-main_freq, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Частота f')
plt.ylabel('|Y\'(f)|')
plt.title('Фурье-образ производной в окрестности основной частоты')
plt.legend()
plt.grid(True, alpha=0.3)

# Сравнение методов в частотной области
plt.subplot(2, 2, 3)
# Теоретический Фурье-образ производной cos(t)
Y_theoretical = 1j * 2 * np.pi * f * Y_fft
plt.plot(f[mask_main], np.abs(Y_theoretical[mask_main]), 'b-', linewidth=2, label='Теоретический')
plt.plot(f[mask_main], np.abs(Y_diff_spectral[mask_main]), 'g-', linewidth=2, label='Вычисленный')
plt.xlabel('Частота f')
plt.ylabel('|Y\'(f)|')
plt.title('Сравнение теоретического и вычисленного Фурье-образов')
plt.legend()
plt.grid(True, alpha=0.3)

# Фазовая характеристика
plt.subplot(2, 2, 4)
phase_original = np.angle(Y_fft[mask_main])
phase_derivative = np.angle(Y_diff_spectral[mask_main])
plt.plot(f[mask_main], phase_original, 'b-', linewidth=2, label='Исходный сигнал')
plt.plot(f[mask_main], phase_derivative, 'g-', linewidth=2, label='Производная')
plt.xlabel('Частота f')
plt.ylabel('Фаза (рад)')
plt.title('Фазовая характеристика')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task1/spectral_differentiation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Вывод статистики
print("Анализ спектрального дифференцирования:")
print("=" * 50)
print(f"Параметры: dt = {dt}, амплитуда шума a = {a}")
print(f"Количество точек: {N}")
print(f"Временной интервал: [{t[0]:.1f}, {t[-1]:.1f}]")
print()
print("Статистика ошибок:")
print(f"Численное дифференцирование:")
print(f"  Среднеквадратичная ошибка: {mse_values[0]:.6f}")
print(f"  Максимальная ошибка: {max_error_values[0]:.6f}")
print()
print(f"Спектральное дифференцирование:")
print(f"  Среднеквадратичная ошибка: {mse_values[1]:.6f}")
print(f"  Максимальная ошибка: {max_error_values[1]:.6f}")
print()
print(f"Улучшение (отношение ошибок): {mse_values[0]/mse_values[1]:.2f}x") 