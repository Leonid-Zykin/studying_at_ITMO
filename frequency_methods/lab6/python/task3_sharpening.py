import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import ndimage
import os

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task3', exist_ok=True)

def create_test_image():
    """Создает тестовое изображение для увеличения резкости"""
    # Создаем изображение 256x256
    size = 256
    image = np.zeros((size, size))
    
    # Добавляем различные объекты с размытыми краями
    # Круг с размытыми краями
    y, x = np.ogrid[:size, :size]
    distance = np.sqrt((x - size//4)**2 + (y - size//4)**2)
    circle = np.exp(-(distance - size//8)**2 / (2 * 5**2))
    image += 0.6 * circle
    
    # Прямоугольник с размытыми краями
    rect_center_y, rect_center_x = size//2, size//2
    rect_width, rect_height = 40, 30
    
    for i in range(size):
        for j in range(size):
            if (abs(i - rect_center_y) < rect_height/2 and 
                abs(j - rect_center_x) < rect_width/2):
                # Размытые края
                edge_dist_y = abs(abs(i - rect_center_y) - rect_height/2)
                edge_dist_x = abs(abs(j - rect_center_x) - rect_width/2)
                edge_dist = min(edge_dist_y, edge_dist_x)
                if edge_dist < 5:
                    image[i, j] += 0.4 * np.exp(-edge_dist**2 / (2 * 2**2))
                else:
                    image[i, j] += 0.4
    
    # Добавляем мелкие детали
    for i in range(0, size, 20):
        for j in range(0, size, 25):
            if np.random.random() > 0.7:
                image[i:i+3, j:j+3] = 0.8
    
    # Добавляем шум
    noise = np.random.normal(0, 0.02, (size, size))
    image += noise
    
    # Ограничиваем значения
    image = np.clip(image, 0, 1)
    
    return image

def create_sharpening_kernel():
    """Создает ядро увеличения резкости"""
    return np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]])

def image_sharpening():
    """Увеличение резкости изображений"""
    
    # Загружаем готовое изображение
    print("Загрузка исходного изображения...")
    try:
        from PIL import Image
        img_path = '../images/task3/original_image.png'
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
    plt.savefig('../images/task3/original_image.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Создаем ядро увеличения резкости
    print("Создание ядра увеличения резкости...")
    sharpening_kernel = create_sharpening_kernel()
    
    # Визуализация ядра
    plt.figure(figsize=(8, 6))
    plt.imshow(sharpening_kernel, cmap='gray', interpolation='nearest')
    plt.title('Ядро увеличения резкости')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../images/task3/sharpening_kernel.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Применяем увеличение резкости с помощью свёртки
    print("Применение увеличения резкости с помощью свёртки...")
    sharpened_conv = ndimage.convolve(original_image, sharpening_kernel, mode='wrap')
    
    # Применяем увеличение резкости несколько раз
    sharpened_conv_2x = ndimage.convolve(sharpened_conv, sharpening_kernel, mode='wrap')
    sharpened_conv_3x = ndimage.convolve(sharpened_conv_2x, sharpening_kernel, mode='wrap')
    
    # Визуализация результатов свёртки
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(sharpened_conv, cmap='gray')
    plt.title('Увеличение резкости (1 раз)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(sharpened_conv_2x, cmap='gray')
    plt.title('Увеличение резкости (2 раза)')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(sharpened_conv_3x, cmap='gray')
    plt.title('Увеличение резкости (3 раза)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task3/convolution_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Применяем увеличение резкости с помощью Фурье-преобразования
    print("Применение увеличения резкости с помощью Фурье-преобразования...")
    
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
    
    kernel_extended[start_h:start_h+k, start_w:start_w+l] = sharpening_kernel
    
    # Фурье-образ ядра
    fft_kernel = fft2(kernel_extended)
    
    # Создаем Фурье-образ изображения того же размера
    fft_original_extended = fft2(original_image, s=(h + k - 1, w + l - 1))
    
    # Поэлементное умножение
    fft_result = fft_original_extended * fft_kernel
    
    # Обратное преобразование
    sharpened_fft_result = np.real(ifft2(fft_result))
    sharpened_fft = sharpened_fft_result[:h, :w]
    
    # Применяем увеличение резкости несколько раз через FFT
    sharpened_fft_2x_result = np.real(ifft2(fft_result * fft_kernel))
    sharpened_fft_2x = sharpened_fft_2x_result[:h, :w]
    
    sharpened_fft_3x_result = np.real(ifft2(fft_result * fft_kernel * fft_kernel))
    sharpened_fft_3x = sharpened_fft_3x_result[:h, :w]
    
    # Визуализация результатов Фурье-метода
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(sharpened_fft, cmap='gray')
    plt.title('Увеличение резкости (FFT, 1 раз)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(sharpened_fft_2x, cmap='gray')
    plt.title('Увеличение резкости (FFT, 2 раза)')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(sharpened_fft_3x, cmap='gray')
    plt.title('Увеличение резкости (FFT, 3 раза)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task3/fft_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сравнение методов
    print("Сравнение методов увеличения резкости...")
    plt.figure(figsize=(20, 15))
    
    # Результаты свёртки
    plt.subplot(3, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(sharpened_conv, cmap='gray')
    plt.title('Свёртка (1 раз)')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(sharpened_conv_2x, cmap='gray')
    plt.title('Свёртка (2 раза)')
    plt.axis('off')
    
    # Результаты FFT
    plt.subplot(3, 3, 4)
    plt.imshow(sharpened_fft, cmap='gray')
    plt.title('FFT (1 раз)')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(sharpened_fft_2x, cmap='gray')
    plt.title('FFT (2 раза)')
    plt.axis('off')
    
    # Разности методов
    diff_1x = np.abs(sharpened_conv - sharpened_fft)
    diff_2x = np.abs(sharpened_conv_2x - sharpened_fft_2x)
    
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
    crop_size = 50
    center_y, center_x = original_image.shape[0]//2, original_image.shape[1]//2
    crop_y = slice(center_y-crop_size//2, center_y+crop_size//2)
    crop_x = slice(center_x-crop_size//2, center_x+crop_size//2)
    
    plt.subplot(3, 3, 8)
    plt.imshow(original_image[crop_y, crop_x], cmap='gray')
    plt.title('Деталь исходного')
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.imshow(sharpened_conv[crop_y, crop_x], cmap='gray')
    plt.title('Деталь увеличенной резкости')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task3/method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Анализ качества увеличения резкости
    print("Анализ качества увеличения резкости...")
    
    # Вычисляем метрики качества
    metrics = {}
    
    # Для однократного применения
    conv_mse_1x = np.mean((original_image - sharpened_conv)**2)
    fft_mse_1x = np.mean((original_image - sharpened_fft)**2)
    method_diff_1x = np.mean((sharpened_conv - sharpened_fft)**2)
    
    # Для двукратного применения
    conv_mse_2x = np.mean((original_image - sharpened_conv_2x)**2)
    fft_mse_2x = np.mean((original_image - sharpened_fft_2x)**2)
    method_diff_2x = np.mean((sharpened_conv_2x - sharpened_fft_2x)**2)
    
    # Для трехкратного применения
    conv_mse_3x = np.mean((original_image - sharpened_conv_3x)**2)
    fft_mse_3x = np.mean((original_image - sharpened_fft_3x)**2)
    method_diff_3x = np.mean((sharpened_conv_3x - sharpened_fft_3x)**2)
    
    metrics = {
        '1x': {'conv_mse': conv_mse_1x, 'fft_mse': fft_mse_1x, 'method_diff': method_diff_1x},
        '2x': {'conv_mse': conv_mse_2x, 'fft_mse': fft_mse_2x, 'method_diff': method_diff_2x},
        '3x': {'conv_mse': conv_mse_3x, 'fft_mse': fft_mse_3x, 'method_diff': method_diff_3x}
    }
    
    # График анализа качества
    plt.figure(figsize=(15, 10))
    
    # MSE для разных методов
    plt.subplot(2, 2, 1)
    iterations = ['1x', '2x', '3x']
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
    plt.savefig('../images/task3/quality_analysis.png', dpi=300, bbox_inches='tight')
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
    plt.title('Спектр ядра увеличения резкости')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(sharpened_conv, cmap='gray')
    plt.title('Результат увеличения резкости')
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
    plt.savefig('../images/task3/spectrum_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Анализ деталей
    print("Анализ деталей изображения...")
    
    # Вычисляем градиенты для оценки резкости
    from scipy.ndimage import sobel
    
    grad_original = np.sqrt(sobel(original_image, axis=0)**2 + sobel(original_image, axis=1)**2)
    grad_sharpened = np.sqrt(sobel(sharpened_conv, axis=0)**2 + sobel(sharpened_conv, axis=1)**2)
    
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
    plt.imshow(sharpened_conv, cmap='gray')
    plt.title('Увеличенная резкость')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(grad_sharpened, cmap='hot')
    plt.title('Градиент увеличенной резкости')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    # Гистограммы градиентов
    plt.hist(grad_original.flatten(), bins=50, alpha=0.7, label='Исходное', density=True)
    plt.hist(grad_sharpened.flatten(), bins=50, alpha=0.7, label='Увеличенная резкость', density=True)
    plt.xlabel('Величина градиента')
    plt.ylabel('Плотность')
    plt.title('Распределение градиентов')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    # Сравнение деталей
    detail_enhancement = grad_sharpened - grad_original
    plt.imshow(detail_enhancement, cmap='hot')
    plt.title('Усиление деталей')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task3/detail_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Вывод результатов
    print("\nРезультаты анализа увеличения резкости:")
    for iteration in ['1x', '2x', '3x']:
        print(f"\n{iteration} применение:")
        print(f"  Свёртка - MSE: {metrics[iteration]['conv_mse']:.6f}")
        print(f"  FFT - MSE: {metrics[iteration]['fft_mse']:.6f}")
        print(f"  Разность методов: {metrics[iteration]['method_diff']:.6f}")
    
    print(f"\nСредняя величина градиента исходного изображения: {np.mean(grad_original):.4f}")
    print(f"Средняя величина градиента увеличенной резкости: {np.mean(grad_sharpened):.4f}")
    print(f"Усиление деталей: {np.mean(grad_sharpened - grad_original):.4f}")
    
    print("\nУвеличение резкости изображений завершено!")
    print("Результаты сохранены в папке images/task3/")

if __name__ == "__main__":
    image_sharpening() 