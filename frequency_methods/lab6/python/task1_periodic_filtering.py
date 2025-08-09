import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
import os

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task1', exist_ok=True)

def create_test_image_with_periodicity():
    """Создает тестовое изображение с периодичностью"""
    # Создаем изображение 512x512
    size = 512
    image = np.zeros((size, size))
    
    # Добавляем горизонтальные полосы (периодичность)
    for i in range(0, size, 20):
        image[i:i+10, :] = 0.8
    
    # Добавляем вертикальные полосы
    for j in range(0, size, 30):
        image[:, j:j+15] = 0.6
    
    # Добавляем диагональные паттерны
    for k in range(0, size, 40):
        for i in range(size):
            for j in range(size):
                if (i + j) % 40 == k:
                    image[i, j] = 0.4
    
    # Добавляем шум
    noise = np.random.normal(0, 0.1, (size, size))
    image += noise
    
    # Ограничиваем значения
    image = np.clip(image, 0, 1)
    
    return image

def periodic_filtering():
    """Фильтрация изображений с периодичностью"""
    
    # Загружаем готовое изображение
    print("Загрузка исходного изображения...")
    try:
        # Загружаем изображение из файла
        img_path = '../images/task1/original_image.png'
        img_pil = Image.open(img_path).convert('L')  # Конвертируем в градации серого
        original_image = np.array(img_pil) / 255.0  # Нормализуем к [0, 1]
        print(f"Загружено изображение размером {original_image.shape}")
    except FileNotFoundError:
        print("Файл original_image.png не найден, создаем тестовое изображение...")
        original_image = create_test_image_with_periodicity()
    
    # Сохраняем исходное изображение
    plt.figure(figsize=(10, 8))
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение с периодичностью')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('../images/task1/original_image.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Шаг 1: Преобразование к вещественному типу и нормализация
    # (у нас уже вещественные значения от 0 до 1)
    image = original_image.copy()
    
    # Шаг 2: Вычисление двумерного Фурье-образа
    print("Вычисление двумерного Фурье-образа...")
    fft_image = fftshift(fft2(image))
    
    # Шаг 3: Разделение на модуль и аргумент
    magnitude = np.abs(fft_image)
    phase = np.angle(fft_image)
    
    # Шаг 4: Логарифмирование модуля и нормализация
    log_magnitude = np.log(magnitude + 1)
    normalized_log_magnitude = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min())
    
    # Сохраняем спектр как изображение
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(normalized_log_magnitude, cmap='gray')
    plt.title('Спектр изображения (логарифм модуля)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(phase, cmap='hsv')
    plt.title('Фаза Фурье-образа')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    # Показываем центральную часть спектра для лучшей видимости
    center = normalized_log_magnitude.shape[0] // 2
    crop_size = 100
    cropped_spectrum = normalized_log_magnitude[center-crop_size:center+crop_size, 
                                               center-crop_size:center+crop_size]
    plt.imshow(cropped_spectrum, cmap='gray')
    plt.title('Центральная часть спектра')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task1/fourier_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Анализ пиков периодичности
    print("Анализ пиков периодичности...")
    
    # Находим пики в спектре
    from scipy.signal import find_peaks
    
    # Анализируем горизонтальные и вертикальные срезы
    center_row = normalized_log_magnitude[normalized_log_magnitude.shape[0]//2, :]
    center_col = normalized_log_magnitude[:, normalized_log_magnitude.shape[1]//2]
    
    # Находим пики
    peaks_row, _ = find_peaks(center_row, height=0.5, distance=10)
    peaks_col, _ = find_peaks(center_col, height=0.5, distance=10)
    
    print(f"Найдено пиков по горизонтали: {len(peaks_row)}")
    print(f"Найдено пиков по вертикали: {len(peaks_col)}")
    
    # График анализа пиков
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(normalized_log_magnitude, cmap='gray')
    plt.title('Полный спектр')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.plot(center_row)
    plt.plot(peaks_row, center_row[peaks_row], 'ro')
    plt.title('Горизонтальный срез спектра')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(center_col)
    plt.plot(peaks_col, center_col[peaks_col], 'ro')
    plt.title('Вертикальный срез спектра')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Показываем пики на спектре
    plt.imshow(normalized_log_magnitude, cmap='gray')
    # Отмечаем пики
    for peak in peaks_row:
        plt.plot(peak, normalized_log_magnitude.shape[0]//2, 'ro', markersize=3)
    for peak in peaks_col:
        plt.plot(normalized_log_magnitude.shape[1]//2, peak, 'ro', markersize=3)
    plt.title('Пики периодичности на спектре')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task1/peak_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Шаг 5: Редактирование спектра для удаления гармоник
    print("Редактирование спектра...")
    
    # Создаем маску для удаления высокочастотных компонентов
    # (симуляция редактирования в программе обработки изображений)
    mask = np.ones_like(magnitude)
    
    # Удаляем высокочастотные компоненты (внешние области)
    center_y, center_x = magnitude.shape[0]//2, magnitude.shape[1]//2
    radius = min(center_y, center_x) // 4  # Радиус для сохранения низких частот
    
    y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Создаем плавную маску
    mask = np.exp(-distance_from_center**2 / (2 * (radius/3)**2))
    
    # Применяем маску к спектру
    filtered_magnitude = magnitude * mask
    filtered_fft = filtered_magnitude * np.exp(1j * phase)
    
    # Шаг 6: Обратное преобразование Фурье
    filtered_image = np.real(ifft2(ifftshift(filtered_fft)))
    
    # Нормализуем результат
    filtered_image = np.clip(filtered_image, 0, 1)
    
    # Сравнение результатов
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(normalized_log_magnitude, cmap='gray')
    plt.title('Спектр исходного изображения')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Маска фильтрации')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Отфильтрованное изображение')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    filtered_log_magnitude = np.log(filtered_magnitude + 1)
    filtered_normalized = (filtered_log_magnitude - filtered_log_magnitude.min()) / (filtered_log_magnitude.max() - filtered_log_magnitude.min())
    plt.imshow(filtered_normalized, cmap='gray')
    plt.title('Спектр отфильтрованного изображения')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    difference = np.abs(original_image - filtered_image)
    plt.imshow(difference, cmap='hot')
    plt.title('Разность изображений')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task1/filtering_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Анализ результатов
    print("Анализ результатов фильтрации:")
    print(f"Средняя яркость исходного изображения: {np.mean(original_image):.4f}")
    print(f"Средняя яркость отфильтрованного изображения: {np.mean(filtered_image):.4f}")
    print(f"Среднеквадратичная ошибка: {np.sqrt(np.mean((original_image - filtered_image)**2)):.4f}")
    print(f"Максимальная разность: {np.max(np.abs(original_image - filtered_image)):.4f}")
    
    # Детальный анализ спектра
    plt.figure(figsize=(15, 10))
    
    # Горизонтальные срезы
    plt.subplot(2, 2, 1)
    plt.plot(center_row, label='Исходный спектр')
    filtered_center_row = filtered_normalized[filtered_normalized.shape[0]//2, :]
    plt.plot(filtered_center_row, label='Отфильтрованный спектр')
    plt.title('Горизонтальный срез спектра')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Вертикальные срезы
    plt.subplot(2, 2, 2)
    plt.plot(center_col, label='Исходный спектр')
    filtered_center_col = filtered_normalized[:, filtered_normalized.shape[1]//2]
    plt.plot(filtered_center_col, label='Отфильтрованный спектр')
    plt.title('Вертикальный срез спектра')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Радиальные профили
    plt.subplot(2, 2, 3)
    angles = np.linspace(0, 2*np.pi, 360)
    radii = np.arange(0, min(center_y, center_x), 5)
    
    original_radial = []
    filtered_radial = []
    
    for r in radii:
        y_coords = center_y + r * np.cos(angles)
        x_coords = center_x + r * np.sin(angles)
        
        y_coords = np.clip(y_coords.astype(int), 0, normalized_log_magnitude.shape[0]-1)
        x_coords = np.clip(x_coords.astype(int), 0, normalized_log_magnitude.shape[1]-1)
        
        original_radial.append(np.mean(normalized_log_magnitude[y_coords, x_coords]))
        filtered_radial.append(np.mean(filtered_normalized[y_coords, x_coords]))
    
    plt.plot(radii, original_radial, label='Исходный спектр')
    plt.plot(radii, filtered_radial, label='Отфильтрованный спектр')
    plt.title('Радиальный профиль спектра')
    plt.xlabel('Радиус')
    plt.ylabel('Средняя амплитуда')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Гистограмма разности
    plt.subplot(2, 2, 4)
    plt.hist(difference.flatten(), bins=50, alpha=0.7, label='Разность')
    plt.title('Гистограмма разности изображений')
    plt.xlabel('Разность')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/task1/detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Фильтрация изображений с периодичностью завершена!")
    print("Результаты сохранены в папке images/task1/")

if __name__ == "__main__":
    periodic_filtering() 