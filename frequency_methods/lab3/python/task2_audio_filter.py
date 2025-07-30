import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
import scipy.io.wavfile as wav
import os

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task2', exist_ok=True)

def load_audio(filename):
    """Загружает аудиофайл и возвращает сигнал и частоту дискретизации"""
    try:
        sample_rate, audio_data = wav.read(filename)
        # Если стерео, берем только первый канал
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        return audio_data, sample_rate
    except Exception as e:
        print(f"Ошибка при загрузке файла {filename}: {e}")
        return None, None

def filter_audio_frequencies(audio_data, sample_rate, freq_ranges_to_remove):
    """
    Фильтрует аудиосигнал, удаляя указанные частотные диапазоны
    
    Args:
        audio_data: исходный аудиосигнал
        sample_rate: частота дискретизации
        freq_ranges_to_remove: список кортежей [(freq1_low, freq1_high), ...]
    
    Returns:
        отфильтрованный сигнал
    """
    # Вычисляем Фурье-образ
    N = len(audio_data)
    U = fftshift(fft(audio_data))
    
    # Создаем массив частот
    freq_step = sample_rate / N
    freqs = np.arange(-sample_rate/2, sample_rate/2, freq_step)
    if len(freqs) != N:
        freqs = np.linspace(-sample_rate/2, sample_rate/2, N, endpoint=False)
    
    # Применяем фильтр
    U_filtered = U.copy()
    for freq_low, freq_high in freq_ranges_to_remove:
        # Обнуляем частоты в заданном диапазоне
        mask = (np.abs(freqs) >= freq_low) & (np.abs(freqs) <= freq_high)
        U_filtered[mask] = 0
    
    # Восстанавливаем сигнал
    audio_filtered = np.real(ifft(ifftshift(U_filtered)))
    
    return audio_filtered, U, U_filtered, freqs

# Загружаем аудиофайл
audio_data, sample_rate = load_audio('../MUHA.wav')

if audio_data is not None:
    print(f"Аудиофайл загружен:")
    print(f"Частота дискретизации: {sample_rate} Гц")
    print(f"Длительность: {len(audio_data) / sample_rate:.2f} секунд")
    print(f"Количество отсчётов: {len(audio_data)}")
    
    # Создаем временную ось
    duration = len(audio_data) / sample_rate
    t = np.linspace(0, duration, len(audio_data), endpoint=False)
    
    # Определяем стратегии фильтрации
    strategies = [
        {
            'name': 'Удаление высоких частот',
            'freq_ranges': [(2000, 22050)],  # убираем частоты выше 2 кГц
            'color': 'green'
        },
        {
            'name': 'Удаление низких частот',
            'freq_ranges': [(0, 50)],  # убираем частоты ниже 50 Гц
            'color': 'orange'
        },
        {
            'name': 'Удаление гармонических помех',
            'freq_ranges': [(950, 1050), (1950, 2050)],  # убираем помехи на 1 и 2 кГц
            'color': 'purple'
        },
        {
            'name': 'Комбинированная фильтрация',
            'freq_ranges': [(0, 50), (2000, 22050), (950, 1050), (1950, 2050)],
            'color': 'red'
        }
    ]
    
    # График 1: Сравнение сигналов во временной области
    plt.figure(figsize=(15, 10))
    
    for i, strategy in enumerate(strategies):
        # Применяем фильтр
        audio_filtered, U, U_filtered, freqs = filter_audio_frequencies(
            audio_data, sample_rate, strategy['freq_ranges']
        )
        
        plt.subplot(2, 2, i+1)
        
        # Показываем только часть сигнала для наглядности
        start_idx = int(0.5 * sample_rate)  # начинаем с 0.5 секунды
        end_idx = int(1.5 * sample_rate)    # заканчиваем на 1.5 секунде
        t_plot = t[start_idx:end_idx]
        
        plt.plot(t_plot, audio_data[start_idx:end_idx], 'b-', alpha=0.7, 
                label='Исходный сигнал', linewidth=1)
        plt.plot(t_plot, audio_filtered[start_idx:end_idx], color=strategy['color'], 
                linewidth=1, label=f'Отфильтрованный ({strategy["name"]})')
        
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.title(f'{strategy["name"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/task2/audio_filter_time_domain.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # График 2: Сравнение Фурье-образов
    plt.figure(figsize=(15, 10))
    
    for i, strategy in enumerate(strategies):
        audio_filtered, U, U_filtered, freqs = filter_audio_frequencies(
            audio_data, sample_rate, strategy['freq_ranges']
        )
        
        plt.subplot(2, 2, i+1)
        
        # Показываем только положительные частоты до 5 кГц для наглядности
        mask_freq = (freqs >= 0) & (freqs <= 5000)
        
        plt.plot(freqs[mask_freq], np.abs(U[mask_freq]), 'b-', alpha=0.7, 
                label='Исходный сигнал', linewidth=1)
        plt.plot(freqs[mask_freq], np.abs(U_filtered[mask_freq]), color=strategy['color'], 
                linewidth=1, label=f'Отфильтрованный')
        
        # Отмечаем диапазоны фильтрации
        for freq_low, freq_high in strategy['freq_ranges']:
            if freq_high <= 5000:  # показываем только в видимом диапазоне
                plt.axvspan(freq_low, freq_high, alpha=0.2, color='red', 
                           label=f'Диапазон фильтрации [{freq_low}, {freq_high}] Гц')
        
        plt.xlabel('Частота (Гц)')
        plt.ylabel('|Ũ(ν)|')
        plt.title(f'{strategy["name"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/task2/audio_filter_freq_domain.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Детальный анализ спектра
    plt.figure(figsize=(15, 10))
    
    # Исходный спектр
    U_original = fftshift(fft(audio_data))
    freqs_original = np.linspace(-sample_rate/2, sample_rate/2, len(audio_data), endpoint=False)
    
    plt.subplot(2, 2, 1)
    mask_freq = (freqs_original >= 0) & (freqs_original <= 1000)
    plt.plot(freqs_original[mask_freq], np.abs(U_original[mask_freq]), 'b-', linewidth=1)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('|Ũ(ν)|')
    plt.title('Исходный спектр (0-1000 Гц)')
    plt.grid(True, alpha=0.3)
    
    # Находим основные частоты
    spectrum = np.abs(U_original[freqs_original >= 0])
    freqs_positive = freqs_original[freqs_original >= 0]
    
    # Ищем пики в спектре
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(spectrum, height=np.max(spectrum)*0.1, distance=50)
    main_frequencies = freqs_positive[peaks]
    
    plt.subplot(2, 2, 2)
    plt.plot(freqs_positive, spectrum, 'b-', linewidth=1)
    plt.plot(main_frequencies, spectrum[peaks], 'ro', markersize=8, label='Основные частоты')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('|Ũ(ν)|')
    plt.title('Основные частоты в сигнале')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Сравнение до и после фильтрации
    best_strategy = strategies[3]  # комбинированная фильтрация
    audio_filtered_best, U_filtered_best, _, _ = filter_audio_frequencies(
        audio_data, sample_rate, best_strategy['freq_ranges']
    )
    
    plt.subplot(2, 2, 3)
    mask_freq = (freqs_original >= 0) & (freqs_original <= 1000)
    plt.plot(freqs_original[mask_freq], np.abs(U_original[mask_freq]), 'b-', alpha=0.7, 
            linewidth=1, label='Исходный сигнал')
    plt.plot(freqs_original[mask_freq], np.abs(U_filtered_best[mask_freq]), 'r-', 
            linewidth=1, label='Отфильтрованный сигнал')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('|Ũ(ν)|')
    plt.title('Сравнение спектров (0-1000 Гц)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Временная область для лучшей стратегии
    plt.subplot(2, 2, 4)
    start_idx = int(0.5 * sample_rate)
    end_idx = int(1.5 * sample_rate)
    t_plot = t[start_idx:end_idx]
    
    plt.plot(t_plot, audio_data[start_idx:end_idx], 'b-', alpha=0.7, 
            linewidth=1, label='Исходный сигнал')
    plt.plot(t_plot, audio_filtered_best[start_idx:end_idx], 'r-', 
            linewidth=1, label='Отфильтрованный сигнал')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.title('Лучшая стратегия фильтрации')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/task2/audio_filter_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохраняем отфильтрованный аудиофайл
    audio_filtered_best = audio_filtered_best.astype(np.int16)
    wav.write('../MUHA_filtered.wav', sample_rate, audio_filtered_best)
    
    print("Анализ аудиофильтрации завершён!")
    print(f"Основные частоты в сигнале: {main_frequencies[:10]} Гц")
    print("Создан отфильтрованный файл: MUHA_filtered.wav")
    
else:
    print("Не удалось загрузить аудиофайл!") 