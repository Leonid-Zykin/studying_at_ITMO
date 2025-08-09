import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import ndimage
import os

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task4', exist_ok=True)

def create_test_image():
    """Создает тестовое изображение для выделения краёв"""
    # Создаем изображение 128x128 (меньший размер для pixel art эффекта)
    size = 128
    image = np.zeros((size, size))
    
    # Добавляем различные объекты с четкими краями
    # Квадрат
    square_size = 30
    start_y, start_x = size//4, size//4
    image[start_y:start_y+square_size, start_x:start_x+square_size] = 0.8
    
    # Круг
    y, x = np.ogrid[:size, :size]
    circle_center_y, circle_center_x = 3*size//4, 3*size//4
    circle_radius = 20
    circle = (x - circle_center_x)**2 + (y - circle_center_y)**2 <= circle_radius**2
    image[circle] = 0.6
    
    # Треугольник
    triangle_center_y, triangle_center_x = size//2, size//2
    triangle_size = 25
    
    for i in range(size):
        for j in range(size):
            # Простой треугольник
            if (abs(i - triangle_center_y) + abs(j - triangle_center_x) < triangle_size and
                i > triangle_center_y - triangle_size//2):
                image[i, j] = 0.4
    
    # Добавляем горизонтальные и вертикальные линии
    for i in range(0, size, 20):
        image[i:i+2, :] = 0.9  # Горизонтальные линии
    
    for j in range(0, size, 25):
        image[:, j:j+2] = 0.9  # Вертикальные линии
    
    # Добавляем диагональные линии
    for k in range(0, size, 15):
        for i in range(size):
            for j in range(size):
                if abs(i - j - k) < 2:
                    image[i, j] = 0.7
    
    # Добавляем мелкие детали (pixel art)
    for i in range(0, size, 8):
        for j in range(0, size, 10):
            if np.random.random() > 0.8:
                image[i:i+3, j:j+3] = 0.5
    
    # Добавляем небольшой шум
    noise = np.random.normal(0, 0.01, (size, size))
    image += noise
    
    # Ограничиваем значения
    image = np.clip(image, 0, 1)
    
    return image

def create_edge_detection_kernel():
    """Создает ядро выделения краёв"""
    return np.array([[-1, -1, -1],
                     [-1, 8, -1],
                     [-1, -1, -1]])

def edge_detection():
    """Выделение краёв изображений"""
    
    # Загружаем готовое изображение
    print("Загрузка исходного изображения...")
    try:
        from PIL import Image
        img_path = '../images/task4/original_image.png'
        img_pil = Image.open(img_path).convert('L')
        original_image = np.array(img_pil) / 255.0
        print(f"Загружено изображение размером {original_image.shape}")
    except FileNotFoundError:
        print("Файл original_image.png не найден, создаем тестовое изображение...")
        original_image = create_test_image()
    
    # Сохраняем исходное изображение
    plt.figure(figsize=(8, 6))
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../images/task4/original_image.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Создаем ядро выделения краёв
    print("Создание ядра выделения краёв...")
    edge_kernel = create_edge_detection_kernel()
    
    # Визуализация ядра
    plt.figure(figsize=(8, 6))
    plt.imshow(edge_kernel, cmap='gray', interpolation='nearest')
    plt.title('Ядро выделения краёв')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../images/task4/edge_kernel.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Применяем выделение краёв с помощью свёртки
    print("Применение выделения краёв с помощью свёртки...")
    edge_conv = ndimage.convolve(original_image, edge_kernel, mode='wrap')
    
    # Нормализуем результат
    edge_conv_normalized = np.clip(edge_conv, 0, 1)
    
    # Применяем выделение краёв несколько раз
    edge_conv_2x = ndimage.convolve(edge_conv_normalized, edge_kernel, mode='wrap')
    edge_conv_2x_normalized = np.clip(edge_conv_2x, 0, 1)
    
    # Визуализация результатов свёртки
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(edge_conv, cmap='gray')
    plt.title('Выделение краёв (1 раз)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(edge_conv_normalized, cmap='gray')
    plt.title('Выделение краёв (нормализованное)')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(edge_conv_2x_normalized, cmap='gray')
    plt.title('Выделение краёв (2 раза)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task4/convolution_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Применяем выделение краёв с помощью Фурье-преобразования
    print("Применение выделения краёв с помощью Фурье-преобразования...")
    
    # Фурье-образ исходного изображения
    fft_original = fft2(original_image)
    
    # Создаем ядро того же размера, что и изображение
    h, w = original_image.shape
    k, l = 3, 3
    
    # Создаем расширенное ядро
    kernel_extended = np.zeros((h + k - 1, w + l - 1))
    
    # Размещаем ядро в центре
    start_h = (h + k - 1 - k) // 2
    start_w = (w + l - 1 - l) // 2
    
    kernel_extended[start_h:start_h+k, start_w:start_w+l] = edge_kernel
    
    # Фурье-образ ядра
    fft_kernel = fft2(kernel_extended)
    
    # Создаем Фурье-образ изображения того же размера
    fft_original_extended = fft2(original_image, s=(h + k - 1, w + l - 1))
    
    # Поэлементное умножение
    fft_result = fft_original_extended * fft_kernel
    
    # Обратное преобразование
    edge_fft_result = np.real(ifft2(fft_result))
    edge_fft = edge_fft_result[:h, :w]
    
    # Нормализуем результат
    edge_fft_normalized = np.clip(edge_fft, 0, 1)
    
    # Применяем выделение краёв несколько раз через FFT
    edge_fft_2x_result = np.real(ifft2(fft_result * fft_kernel))
    edge_fft_2x = edge_fft_2x_result[:h, :w]
    edge_fft_2x_normalized = np.clip(edge_fft_2x, 0, 1)
    
    # Визуализация результатов Фурье-метода
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(edge_fft, cmap='gray')
    plt.title('Выделение краёв (FFT, 1 раз)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(edge_fft_normalized, cmap='gray')
    plt.title('Выделение краёв (FFT, нормализованное)')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(edge_fft_2x_normalized, cmap='gray')
    plt.title('Выделение краёв (FFT, 2 раза)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task4/fft_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сравнение методов
    print("Сравнение методов выделения краёв...")
    plt.figure(figsize=(20, 15))
    
    # Результаты свёртки
    plt.subplot(3, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(edge_conv_normalized, cmap='gray')
    plt.title('Свёртка (1 раз)')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(edge_conv_2x_normalized, cmap='gray')
    plt.title('Свёртка (2 раза)')
    plt.axis('off')
    
    # Результаты FFT
    plt.subplot(3, 3, 4)
    plt.imshow(edge_fft_normalized, cmap='gray')
    plt.title('FFT (1 раз)')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(edge_fft_2x_normalized, cmap='gray')
    plt.title('FFT (2 раза)')
    plt.axis('off')
    
    # Разности методов
    diff_1x = np.abs(edge_conv_normalized - edge_fft_normalized)
    diff_2x = np.abs(edge_conv_2x_normalized - edge_fft_2x_normalized)
    
    plt.subplot(3, 3, 6)
    plt.imshow(diff_1x, cmap='hot')
    plt.title('Разность методов (1 раз)')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(3, 3, 7)
    plt.imshow(diff_2x, cmap='hot')
    plt.title('Разность методов (2 раза)')
    plt.colorbar()
    plt.axis('off')
    
    # Детали изображений
    crop_size = 40
    center_y, center_x = original_image.shape[0]//2, original_image.shape[1]//2
    crop_y = slice(center_y-crop_size//2, center_y+crop_size//2)
    crop_x = slice(center_x-crop_size//2, center_x+crop_size//2)
    
    plt.subplot(3, 3, 8)
    plt.imshow(original_image[crop_y, crop_x], cmap='gray')
    plt.title('Деталь исходного')
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.imshow(edge_conv_normalized[crop_y, crop_x], cmap='gray')
    plt.title('Деталь выделенных краёв')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task4/method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Анализ качества выделения краёв
    print("Анализ качества выделения краёв...")
    
    # Вычисляем метрики качества
    metrics = {}
    
    # Для однократного применения
    conv_mse_1x = np.mean((original_image - edge_conv_normalized)**2)
    fft_mse_1x = np.mean((original_image - edge_fft_normalized)**2)
    method_diff_1x = np.mean((edge_conv_normalized - edge_fft_normalized)**2)
    
    # Для двукратного применения
    conv_mse_2x = np.mean((original_image - edge_conv_2x_normalized)**2)
    fft_mse_2x = np.mean((original_image - edge_fft_2x_normalized)**2)
    method_diff_2x = np.mean((edge_conv_2x_normalized - edge_fft_2x_normalized)**2)
    
    metrics = {
        '1x': {'conv_mse': conv_mse_1x, 'fft_mse': fft_mse_1x, 'method_diff': method_diff_1x},
        '2x': {'conv_mse': conv_mse_2x, 'fft_mse': fft_mse_2x, 'method_diff': method_diff_2x}
    }
    
    # График анализа качества
    plt.figure(figsize=(15, 10))
    
    # MSE для разных методов
    plt.subplot(2, 2, 1)
    iterations = ['1x', '2x']
    conv_mses = [metrics[it]['conv_mse'] for it in iterations]
    fft_mses = [metrics[it]['fft_mse'] for it in iterations]
    
    plt.plot(iterations, conv_mses, 'bo-', label='Свёртка')
    plt.plot(iterations, fft_mses, 'ro-', label='FFT')
    plt.xlabel('Количество применений')
    plt.ylabel('MSE')
    plt.title('Среднеквадратичная ошибка')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Разность между методами
    plt.subplot(2, 2, 2)
    method_diffs = [metrics[it]['method_diff'] for it in iterations]
    
    plt.plot(iterations, method_diffs, 'go-')
    plt.xlabel('Количество применений')
    plt.ylabel('Разность методов')
    plt.title('Разность между свёрткой и FFT')
    plt.grid(True, alpha=0.3)
    
    # Сравнение с исходным изображением
    plt.subplot(2, 2, 3)
    plt.plot(iterations, conv_mses, 'bo-', label='Свёртка')
    plt.plot(iterations, fft_mses, 'ro-', label='FFT')
    plt.xlabel('Количество применений')
    plt.ylabel('MSE относительно исходного')
    plt.title('Сравнение с исходным изображением')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Детальный анализ для 1x
    plt.subplot(2, 2, 4)
    methods = ['Исходное', 'Свёртка (1x)', 'FFT (1x)']
    mses = [0, metrics['1x']['conv_mse'], metrics['1x']['fft_mse']]
    
    plt.bar(methods, mses)
    plt.ylabel('MSE')
    plt.title('Сравнение методов для однократного применения')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/task4/quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Детальный анализ спектров
    print("Детальный анализ спектров...")
    
    # Фурье-образы
    fft_original_shifted = fftshift(fft2(original_image))
    fft_kernel_shifted = fftshift(fft2(kernel_extended))
    
    # Визуализация спектров
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    log_magnitude_original = np.log(np.abs(fft_original_shifted) + 1)
    plt.imshow(log_magnitude_original, cmap='gray')
    plt.title('Спектр исходного изображения')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    log_magnitude_kernel = np.log(np.abs(fft_kernel_shifted) + 1)
    plt.imshow(log_magnitude_kernel, cmap='gray')
    plt.title('Спектр ядра выделения краёв')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(edge_conv_normalized, cmap='gray')
    plt.title('Результат выделения краёв')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    fft_result_shifted = fftshift(fft_result)
    log_magnitude_result = np.log(np.abs(fft_result_shifted) + 1)
    plt.imshow(log_magnitude_result, cmap='gray')
    plt.title('Спектр результата')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    # Сравнение профилей
    center_y, center_x = log_magnitude_original.shape[0]//2, log_magnitude_original.shape[1]//2
    
    # Горизонтальный срез
    original_slice = log_magnitude_original[center_y, :]
    kernel_slice = log_magnitude_kernel[center_y, :]
    result_slice = log_magnitude_result[center_y, :]
    
    plt.plot(original_slice, 'b-', label='Исходный', alpha=0.7)
    plt.plot(kernel_slice, 'r-', label='Ядро', alpha=0.7)
    plt.plot(result_slice, 'g-', label='Результат', alpha=0.7)
    plt.title('Горизонтальный срез спектров')
    plt.xlabel('Частота')
    plt.ylabel('Логарифм амплитуды')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/task4/spectrum_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Анализ краёв
    print("Анализ краёв изображения...")
    
    # Вычисляем градиенты для сравнения
    from scipy.ndimage import sobel
    
    grad_original = np.sqrt(sobel(original_image, axis=0)**2 + sobel(original_image, axis=1)**2)
    grad_edge_detected = np.sqrt(sobel(edge_conv_normalized, axis=0)**2 + sobel(edge_conv_normalized, axis=1)**2)
    
    # Визуализация градиентов
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(grad_original, cmap='hot')
    plt.title('Градиент исходного изображения')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(edge_conv_normalized, cmap='gray')
    plt.title('Выделенные края')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(grad_edge_detected, cmap='hot')
    plt.title('Градиент выделенных краёв')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    # Гистограммы градиентов
    plt.hist(grad_original.flatten(), bins=50, alpha=0.7, label='Исходное', density=True)
    plt.hist(grad_edge_detected.flatten(), bins=50, alpha=0.7, label='Выделенные края', density=True)
    plt.xlabel('Величина градиента')
    plt.ylabel('Плотность')
    plt.title('Распределение градиентов')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    # Сравнение деталей
    edge_enhancement = grad_edge_detected - grad_original
    plt.imshow(edge_enhancement, cmap='hot')
    plt.title('Усиление краёв')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task4/edge_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Дополнительный анализ краёв
    print("Дополнительный анализ краёв...")
    
    # Находим пороговые значения для выделения краёв
    edge_threshold = np.percentile(edge_conv, 90)
    binary_edges = edge_conv > edge_threshold
    
    # Анализ направлений краёв
    from scipy.ndimage import sobel
    
    grad_x = sobel(original_image, axis=1)
    grad_y = sobel(original_image, axis=0)
    edge_direction = np.arctan2(grad_y, grad_x)
    
    # Визуализация дополнительного анализа
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(edge_conv, cmap='gray')
    plt.title('Выделение краёв (сырой результат)')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(binary_edges, cmap='gray')
    plt.title('Бинарные края')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(edge_direction, cmap='hsv')
    plt.title('Направления градиентов')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    # Гистограмма направлений
    plt.hist(edge_direction.flatten(), bins=50, alpha=0.7)
    plt.xlabel('Направление (радианы)')
    plt.ylabel('Частота')
    plt.title('Распределение направлений градиентов')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    # Сравнение с другими методами выделения краёв
    from scipy.ndimage import laplace
    laplace_edges = np.abs(laplace(original_image))
    plt.imshow(laplace_edges, cmap='gray')
    plt.title('Оператор Лапласа')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task4/additional_edge_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Вывод результатов
    print("\nРезультаты анализа выделения краёв:")
    for iteration in ['1x', '2x']:
        print(f"\n{iteration} применение:")
        print(f"  Свёртка - MSE: {metrics[iteration]['conv_mse']:.6f}")
        print(f"  FFT - MSE: {metrics[iteration]['fft_mse']:.6f}")
        print(f"  Разность методов: {metrics[iteration]['method_diff']:.6f}")
    
    print(f"\nСредняя величина градиента исходного изображения: {np.mean(grad_original):.4f}")
    print(f"Средняя величина градиента выделенных краёв: {np.mean(grad_edge_detected):.4f}")
    print(f"Усиление краёв: {np.mean(grad_edge_detected - grad_original):.4f}")
    print(f"Пороговое значение для выделения краёв: {edge_threshold:.4f}")
    print(f"Количество пикселей краёв: {np.sum(binary_edges)}")
    
    print("\nВыделение краёв изображений завершено!")
    print("Результаты сохранены в папке images/task4/")

if __name__ == "__main__":
    edge_detection() 