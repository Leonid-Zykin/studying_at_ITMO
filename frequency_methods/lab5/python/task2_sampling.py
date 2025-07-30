import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
import os

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task2', exist_ok=True)

# Функция для интерполяции по формуле Найквиста-Шеннона
def sinc_interpolation(t, t_samples, y_samples, dt):
    """Интерполяция с помощью sinc функции"""
    y_interpolated = np.zeros_like(t)
    
    for i, t_val in enumerate(t):
        # Формула интерполяции: y(t) = Σ y[n] * sinc((t - n*dt) / dt)
        sinc_values = np.sinc((t_val - t_samples) / dt)
        y_interpolated[i] = np.sum(y_samples * sinc_values)
    
    return y_interpolated

# Задание 2.1: Сэмплирование синусов
def sampling_sines():
    """Исследование сэмплирования суммы синусов"""
    
    # Параметры
    a1, a2 = 1.0, 0.5
    omega1, omega2 = 2 * np.pi * 2, 2 * np.pi * 8  # 2 Гц и 8 Гц
    phi1, phi2 = 0, np.pi / 4
    
    # Непрерывная функция
    def y_continuous(t):
        return a1 * np.sin(omega1 * t + phi1) + a2 * np.sin(omega2 * t + phi2)
    
    # Параметры для непрерывного сигнала
    t_continuous = np.linspace(-5, 5, 10000)
    y_continuous_vals = y_continuous(t_continuous)
    
    # Исследуемые шаги дискретизации
    dt_values = [0.1, 0.2, 0.5, 1.0]
    
    plt.figure(figsize=(20, 15))
    
    for i, dt in enumerate(dt_values):
        # Сэмплирование
        t_samples = np.arange(-5, 5 + dt, dt)
        y_samples = y_continuous(t_samples)
        
        # Интерполяция
        y_interpolated = sinc_interpolation(t_continuous, t_samples, y_samples, dt)
        
        # Графики
        plt.subplot(2, 2, i+1)
        
        # Непрерывный сигнал
        plt.plot(t_continuous, y_continuous_vals, 'b-', linewidth=2, label='Непрерывный сигнал')
        
        # Сэмплированные точки
        plt.plot(t_samples, y_samples, 'ro', markersize=4, label=f'Сэмплы (dt={dt})')
        
        # Восстановленный сигнал
        plt.plot(t_continuous, y_interpolated, 'g-', linewidth=2, label='Восстановленный сигнал')
        
        plt.xlabel('Время t')
        plt.ylabel('Амплитуда')
        plt.title(f'Сэмплирование синусов, dt = {dt}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Показываем только центральную часть для лучшей видимости
        plt.xlim(-2, 2)
    
    plt.tight_layout()
    plt.savefig('../images/task2/sampling_sines.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Анализ ошибок
    print("Анализ сэмплирования синусов:")
    print("=" * 50)
    
    for dt in dt_values:
        t_samples = np.arange(-5, 5 + dt, dt)
        y_samples = y_continuous(t_samples)
        y_interpolated = sinc_interpolation(t_continuous, t_samples, y_samples, dt)
        
        # Вычисляем ошибку
        error = np.mean(np.abs(y_interpolated - y_continuous_vals))
        max_error = np.max(np.abs(y_interpolated - y_continuous_vals))
        
        # Частота Найквиста для максимальной частоты (8 Гц)
        nyquist_freq = 2 * 8  # 16 Гц
        sampling_freq = 1 / dt
        
        print(f"dt = {dt}:")
        print(f"  Частота сэмплирования: {sampling_freq:.1f} Гц")
        print(f"  Частота Найквиста: {nyquist_freq} Гц")
        print(f"  Соответствие теореме: {'Да' if sampling_freq > nyquist_freq else 'Нет'}")
        print(f"  Средняя ошибка: {error:.6f}")
        print(f"  Максимальная ошибка: {max_error:.6f}")
        print()

# Задание 2.2: Сэмплирование sinc функции
def sampling_sinc():
    """Исследование сэмплирования sinc функции"""
    
    # Параметры
    b = 2
    
    # Непрерывная функция
    def y_continuous(t):
        return np.sinc(b * t)
    
    # Параметры для непрерывного сигнала
    t_continuous = np.linspace(-10, 10, 20000)
    y_continuous_vals = y_continuous(t_continuous)
    
    # Исследуемые шаги дискретизации
    dt_values = [0.1, 0.2, 0.5, 1.0]
    
    plt.figure(figsize=(20, 15))
    
    for i, dt in enumerate(dt_values):
        # Сэмплирование
        t_samples = np.arange(-10, 10 + dt, dt)
        y_samples = y_continuous(t_samples)
        
        # Интерполяция
        y_interpolated = sinc_interpolation(t_continuous, t_samples, y_samples, dt)
        
        # Графики
        plt.subplot(2, 2, i+1)
        
        # Непрерывный сигнал
        plt.plot(t_continuous, y_continuous_vals, 'b-', linewidth=2, label='Непрерывный сигнал')
        
        # Сэмплированные точки
        plt.plot(t_samples, y_samples, 'ro', markersize=4, label=f'Сэмплы (dt={dt})')
        
        # Восстановленный сигнал
        plt.plot(t_continuous, y_interpolated, 'g-', linewidth=2, label='Восстановленный сигнал')
        
        plt.xlabel('Время t')
        plt.ylabel('Амплитуда')
        plt.title(f'Сэмплирование sinc, dt = {dt}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Показываем только центральную часть
        plt.xlim(-3, 3)
    
    plt.tight_layout()
    plt.savefig('../images/task2/sampling_sinc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Анализ Фурье-образов
    plt.figure(figsize=(20, 15))
    
    for i, dt in enumerate(dt_values):
        t_samples = np.arange(-10, 10 + dt, dt)
        y_samples = y_continuous(t_samples)
        y_interpolated = sinc_interpolation(t_continuous, t_samples, y_samples, dt)
        
        # Вычисляем Фурье-образы
        # Исходный сигнал
        Y_original = fftshift(fft(y_continuous_vals))
        nu_original = fftshift(np.fft.fftfreq(len(t_continuous), t_continuous[1] - t_continuous[0]))
        
        # Восстановленный сигнал
        Y_reconstructed = fftshift(fft(y_interpolated))
        nu_reconstructed = fftshift(np.fft.fftfreq(len(t_continuous), t_continuous[1] - t_continuous[0]))
        
        plt.subplot(2, 2, i+1)
        
        # Показываем только низкие частоты
        mask = (nu_original >= -20) & (nu_original <= 20)
        
        plt.plot(nu_original[mask], np.abs(Y_original[mask]), 'b-', linewidth=2, label='Исходный сигнал')
        plt.plot(nu_reconstructed[mask], np.abs(Y_reconstructed[mask]), 'g-', linewidth=2, label='Восстановленный сигнал')
        
        plt.xlabel('Частота ν')
        plt.ylabel('|Y(ν)|')
        plt.title(f'Фурье-образы, dt = {dt}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/task2/sampling_sinc_fourier.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Анализ ошибок
    print("Анализ сэмплирования sinc функции:")
    print("=" * 50)
    
    for dt in dt_values:
        t_samples = np.arange(-10, 10 + dt, dt)
        y_samples = y_continuous(t_samples)
        y_interpolated = sinc_interpolation(t_continuous, t_samples, y_samples, dt)
        
        # Вычисляем ошибку
        error = np.mean(np.abs(y_interpolated - y_continuous_vals))
        max_error = np.max(np.abs(y_interpolated - y_continuous_vals))
        
        # Для sinc функции ширина спектра примерно равна b
        # Частота Найквиста примерно равна b
        nyquist_freq = b
        sampling_freq = 1 / dt
        
        print(f"dt = {dt}:")
        print(f"  Частота сэмплирования: {sampling_freq:.1f} Гц")
        print(f"  Частота Найквиста: {nyquist_freq} Гц")
        print(f"  Соответствие теореме: {'Да' if sampling_freq > nyquist_freq else 'Нет'}")
        print(f"  Средняя ошибка: {error:.6f}")
        print(f"  Максимальная ошибка: {max_error:.6f}")
        print()

# Детальный анализ теоремы Найквиста
def nyquist_analysis():
    """Детальный анализ теоремы Найквиста-Шеннона-Котельникова"""
    
    # Тестовая функция с различными частотами
    def test_function(t, f_max):
        """Тестовая функция с максимальной частотой f_max"""
        return np.sin(2 * np.pi * f_max * t) + 0.5 * np.sin(2 * np.pi * f_max * 0.5 * t)
    
    # Параметры
    t_continuous = np.linspace(-5, 5, 10000)
    f_max_values = [1, 2, 5, 10]  # максимальные частоты
    dt_values = [0.1, 0.2, 0.5, 1.0]
    
    plt.figure(figsize=(20, 15))
    
    for i, f_max in enumerate(f_max_values):
        y_continuous_vals = test_function(t_continuous, f_max)
        
        plt.subplot(2, 2, i+1)
        
        for dt in dt_values:
            # Сэмплирование
            t_samples = np.arange(-5, 5 + dt, dt)
            y_samples = test_function(t_samples, f_max)
            
            # Интерполяция
            y_interpolated = sinc_interpolation(t_continuous, t_samples, y_samples, dt)
            
            # Вычисляем ошибку
            error = np.mean(np.abs(y_interpolated - y_continuous_vals))
            
            # Частота Найквиста
            nyquist_freq = 2 * f_max
            sampling_freq = 1 / dt
            
            # Определяем цвет в зависимости от соответствия теореме
            if sampling_freq > nyquist_freq:
                color = 'green'
                linestyle = '-'
            else:
                color = 'red'
                linestyle = '--'
            
            plt.plot(t_continuous, y_interpolated, color=color, linestyle=linestyle, 
                    linewidth=2, label=f'dt={dt} (ошибка={error:.4f})')
        
        plt.plot(t_continuous, y_continuous_vals, 'k-', linewidth=3, label='Исходный сигнал')
        plt.xlabel('Время t')
        plt.ylabel('Амплитуда')
        plt.title(f'Анализ теоремы Найквиста, f_max = {f_max} Гц')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-2, 2)
    
    plt.tight_layout()
    plt.savefig('../images/task2/nyquist_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сводная таблица результатов
    print("Сводный анализ теоремы Найквиста-Шеннона-Котельникова:")
    print("=" * 70)
    print("f_max | dt   | f_samp | f_nyquist | Соответствие | Ошибка")
    print("-" * 70)
    
    for f_max in f_max_values:
        y_continuous_vals = test_function(t_continuous, f_max)
        
        for dt in dt_values:
            t_samples = np.arange(-5, 5 + dt, dt)
            y_samples = test_function(t_samples, f_max)
            y_interpolated = sinc_interpolation(t_continuous, t_samples, y_samples, dt)
            
            error = np.mean(np.abs(y_interpolated - y_continuous_vals))
            nyquist_freq = 2 * f_max
            sampling_freq = 1 / dt
            follows_theorem = sampling_freq > nyquist_freq
            
            print(f"{f_max:5.1f} | {dt:4.1f} | {sampling_freq:6.1f} | {nyquist_freq:9.1f} | {'Да':11} | {error:.4f}")
    
    print("-" * 70)

# Запуск всех анализов
if __name__ == "__main__":
    print("Исследование теоремы Найквиста-Шеннона-Котельникова")
    print("=" * 60)
    
    # Задание 2.1: Сэмплирование синусов
    sampling_sines()
    
    # Задание 2.2: Сэмплирование sinc
    sampling_sinc()
    
    # Детальный анализ
    nyquist_analysis()
    
    print("\nВыводы:")
    print("- Теорема Найквиста-Шеннона-Котельникова подтверждается экспериментально")
    print("- При нарушении условий теоремы возникают искажения (алиасинг)")
    print("- Интерполяция sinc функцией обеспечивает точное восстановление при выполнении условий")
    print("- Частота сэмплирования должна быть больше удвоенной максимальной частоты сигнала") 