import numpy as np
import scipy.io.wavfile as wav
import os

# Создаем тестовый аудиофайл с голосом и шумами
def create_test_audio():
    # Параметры аудио
    sample_rate = 44100  # частота дискретизации
    duration = 3.0  # длительность в секундах
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Создаем "голос" - комбинация низких частот
    voice_freqs = [150, 300, 450, 600]  # основные частоты голоса
    voice = np.zeros_like(t)
    for freq in voice_freqs:
        voice += 0.3 * np.sin(2 * np.pi * freq * t)
    
    # Добавляем модуляцию для имитации речи
    modulation = 0.5 * np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 8 * t)
    voice *= (1 + 0.2 * modulation)
    
    # Создаем шумы
    # Высокочастотный шум
    high_freq_noise = 0.2 * np.random.randn(len(t))
    
    # Низкочастотный гул
    low_freq_hum = 0.3 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 100 * t)
    
    # Гармонические помехи
    harmonic_noise = 0.1 * np.sin(2 * np.pi * 1000 * t) + 0.1 * np.sin(2 * np.pi * 2000 * t)
    
    # Комбинируем сигнал
    audio_signal = voice + high_freq_noise + low_freq_hum + harmonic_noise
    
    # Нормализуем сигнал
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    
    # Конвертируем в 16-битный формат
    audio_signal = (audio_signal * 32767).astype(np.int16)
    
    # Сохраняем файл
    wav.write('../MUHA.wav', sample_rate, audio_signal)
    
    print("Тестовый аудиофайл MUHA.wav создан!")
    print(f"Частота дискретизации: {sample_rate} Гц")
    print(f"Длительность: {duration} секунд")
    print("Содержит: голос (150-600 Гц) + высокочастотный шум + низкочастотный гул + гармонические помехи")

if __name__ == "__main__":
    create_test_audio() 