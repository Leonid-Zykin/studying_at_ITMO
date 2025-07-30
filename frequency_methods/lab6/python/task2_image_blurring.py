import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import ndimage
import os

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task2', exist_ok=True)

def create_test_image():
    """Создает тестовое изображение для размытия"""
    # Создаем изображение 256x256
    size = 256
    image = np.zeros((size, size))
    
    # Добавляем различные объекты
    # Круг
    y, x = np.ogrid[:size, :size]
    circle = (x - size//4)**2 + (y - size//4)**2 <= (size//8)**2
    image[circle] = 0.8
    
    # Прямоугольник
    image[size//2-30:size//2+30, size//2-20:size//2+20] = 0.6
    
    # Текстоподобные структуры
    for i in range(0, size, 40):
        image[i:i+5, :] = 0.4
    
    for j in range(0, size, 50):
        image[:, j:j+3] = 0.3
    
    # Добавляем шум
    noise = np.random.normal(0, 0.05, (size, size))
    image += noise
    
    # Ограничиваем значения
    image = np.clip(image, 0, 1)
    
    return image

def create_block_kernel(n):
    """Создает ядро блочного размытия"""
    return np.ones((n, n)) / (n * n)

def create_gaussian_kernel(n):
    """Создает ядро гауссовского размытия"""
    kernel = np.zeros((n, n))
    center = (n - 1) / 2
    
    for i in range(n):
        for j in range(n):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-9 / (n * n) * (x * x + y * y))
    
    # Нормализуем ядро
    kernel = kernel / np.sum(kernel)
    
    return kernel

def image_blurring():
    """Размытие изображений"""
    
    # Создаем тестовое изображение
    print("Создание тестового изображения...")
    original_image = create_test_image()
    
    # Сохраняем исходное изображение
    plt.figure(figsize=(8, 6))
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../images/task2/original_image.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Значения n для исследования
    n_values = [3, 5, 7]
    
    # Создаем ядра размытия
    print("Создание ядер размытия...")
    block_kernels = {}
    gaussian_kernels = {}
    
    for n in n_values:
        block_kernels[n] = create_block_kernel(n)
        gaussian_kernels[n] = create_gaussian_kernel(n)
    
    # Визуализация ядер
    plt.figure(figsize=(15, 10))
    
    for i, n in enumerate(n_values):
        plt.subplot(2, 3, i+1)
        plt.imshow(block_kernels[n], cmap='gray')
        plt.title(f'Блочное ядро n={n}')
        plt.axis('off')
        
        plt.subplot(2, 3, i+4)
        plt.imshow(gaussian_kernels[n], cmap='gray')
        plt.title(f'Гауссовское ядро n={n}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task2/blur_kernels.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Применяем размытие с помощью свёртки
    print("Применение размытия с помощью свёртки...")
    block_results = {}
    gaussian_results = {}
    
    for n in n_values:
        # Блочное размытие
        block_results[n] = ndimage.convolve(original_image, block_kernels[n], mode='constant', cval=0)
        
        # Гауссовское размытие
        gaussian_results[n] = ndimage.convolve(original_image, gaussian_kernels[n], mode='constant', cval=0)
    
    # Визуализация результатов свёртки
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 4, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    for i, n in enumerate(n_values):
        plt.subplot(2, 4, i+2)
        plt.imshow(block_results[n], cmap='gray')
        plt.title(f'Блочное размытие n={n}')
        plt.axis('off')
        
        plt.subplot(2, 4, i+5)
        plt.imshow(gaussian_results[n], cmap='gray')
        plt.title(f'Гауссовское размытие n={n}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task2/convolution_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Применяем размытие с помощью Фурье-преобразования
    print("Применение размытия с помощью Фурье-преобразования...")
    block_fft_results = {}
    gaussian_fft_results = {}
    
    # Фурье-образ исходного изображения
    fft_original = fft2(original_image)
    
    for n in n_values:
        # Создаем ядра того же размера, что и изображение
        h, w = original_image.shape
        k, l = n, n
        
        # Создаем расширенные ядра
        block_kernel_extended = np.zeros((h + k - 1, w + l - 1))
        gaussian_kernel_extended = np.zeros((h + k - 1, w + l - 1))
        
        # Размещаем ядра в центре
        start_h = (h + k - 1 - k) // 2
        start_w = (w + l - 1 - l) // 2
        
        block_kernel_extended[start_h:start_h+k, start_w:start_w+l] = block_kernels[n]
        gaussian_kernel_extended[start_h:start_h+k, start_w:start_w+l] = gaussian_kernels[n]
        
        # Фурье-образы ядер
        fft_block_kernel = fft2(block_kernel_extended)
        fft_gaussian_kernel = fft2(gaussian_kernel_extended)
        
        # Создаем Фурье-образ изображения того же размера
        fft_original_extended = fft2(original_image, s=(h + k - 1, w + l - 1))
        
        # Поэлементное умножение
        fft_block_result = fft_original_extended * fft_block_kernel
        fft_gaussian_result = fft_original_extended * fft_gaussian_kernel
        
        # Обратное преобразование
        block_fft_result = np.real(ifft2(fft_block_result))
        gaussian_fft_result = np.real(ifft2(fft_gaussian_result))
        
        # Обрезаем до исходного размера
        block_fft_results[n] = block_fft_result[:h, :w]
        gaussian_fft_results[n] = gaussian_fft_result[:h, :w]
    
    # Визуализация результатов Фурье-метода
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 4, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    for i, n in enumerate(n_values):
        plt.subplot(2, 4, i+2)
        plt.imshow(block_fft_results[n], cmap='gray')
        plt.title(f'Блочное размытие (FFT) n={n}')
        plt.axis('off')
        
        plt.subplot(2, 4, i+5)
        plt.imshow(gaussian_fft_results[n], cmap='gray')
        plt.title(f'Гауссовское размытие (FFT) n={n}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task2/fft_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сравнение методов
    print("Сравнение методов размытия...")
    plt.figure(figsize=(20, 15))
    
    for i, n in enumerate(n_values):
        # Блочное размытие
        plt.subplot(3, 4, i+1)
        plt.imshow(block_results[n], cmap='gray')
        plt.title(f'Блочное размытие (свёртка) n={n}')
        plt.axis('off')
        
        plt.subplot(3, 4, i+5)
        plt.imshow(block_fft_results[n], cmap='gray')
        plt.title(f'Блочное размытие (FFT) n={n}')
        plt.axis('off')
        
        # Разность методов
        diff_block = np.abs(block_results[n] - block_fft_results[n])
        plt.subplot(3, 4, i+9)
        plt.imshow(diff_block, cmap='hot')
        plt.title(f'Разность блочного n={n}')
        plt.colorbar()
        plt.axis('off')
        
        # Гауссовское размытие
        plt.subplot(3, 4, i+2)
        plt.imshow(gaussian_results[n], cmap='gray')
        plt.title(f'Гауссовское размытие (свёртка) n={n}')
        plt.axis('off')
        
        plt.subplot(3, 4, i+6)
        plt.imshow(gaussian_fft_results[n], cmap='gray')
        plt.title(f'Гауссовское размытие (FFT) n={n}')
        plt.axis('off')
        
        # Разность методов
        diff_gaussian = np.abs(gaussian_results[n] - gaussian_fft_results[n])
        plt.subplot(3, 4, i+10)
        plt.imshow(diff_gaussian, cmap='hot')
        plt.title(f'Разность гауссовского n={n}')
        plt.colorbar()
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task2/method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Анализ качества размытия
    print("Анализ качества размытия...")
    
    # Вычисляем метрики качества
    metrics = {}
    
    for n in n_values:
        # Для блочного размытия
        block_conv_mse = np.mean((original_image - block_results[n])**2)
        block_fft_mse = np.mean((original_image - block_fft_results[n])**2)
        block_method_diff = np.mean((block_results[n] - block_fft_results[n])**2)
        
        # Для гауссовского размытия
        gaussian_conv_mse = np.mean((original_image - gaussian_results[n])**2)
        gaussian_fft_mse = np.mean((original_image - gaussian_fft_results[n])**2)
        gaussian_method_diff = np.mean((gaussian_results[n] - gaussian_fft_results[n])**2)
        
        metrics[n] = {
            'block_conv_mse': block_conv_mse,
            'block_fft_mse': block_fft_mse,
            'block_method_diff': block_method_diff,
            'gaussian_conv_mse': gaussian_conv_mse,
            'gaussian_fft_mse': gaussian_fft_mse,
            'gaussian_method_diff': gaussian_method_diff
        }
    
    # График анализа качества
    plt.figure(figsize=(15, 10))
    
    # MSE для разных методов
    plt.subplot(2, 2, 1)
    n_list = list(metrics.keys())
    block_conv_mses = [metrics[n]['block_conv_mse'] for n in n_list]
    block_fft_mses = [metrics[n]['block_fft_mse'] for n in n_list]
    gaussian_conv_mses = [metrics[n]['gaussian_conv_mse'] for n in n_list]
    gaussian_fft_mses = [metrics[n]['gaussian_fft_mse'] for n in n_list]
    
    plt.plot(n_list, block_conv_mses, 'bo-', label='Блочное (свёртка)')
    plt.plot(n_list, block_fft_mses, 'bs--', label='Блочное (FFT)')
    plt.plot(n_list, gaussian_conv_mses, 'ro-', label='Гауссовское (свёртка)')
    plt.plot(n_list, gaussian_fft_mses, 'rs--', label='Гауссовское (FFT)')
    plt.xlabel('Размер ядра n')
    plt.ylabel('MSE')
    plt.title('Среднеквадратичная ошибка')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Разность между методами
    plt.subplot(2, 2, 2)
    block_method_diffs = [metrics[n]['block_method_diff'] for n in n_list]
    gaussian_method_diffs = [metrics[n]['gaussian_method_diff'] for n in n_list]
    
    plt.plot(n_list, block_method_diffs, 'bo-', label='Блочное размытие')
    plt.plot(n_list, gaussian_method_diffs, 'ro-', label='Гауссовское размытие')
    plt.xlabel('Размер ядра n')
    plt.ylabel('Разность методов')
    plt.title('Разность между свёрткой и FFT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Сравнение типов размытия
    plt.subplot(2, 2, 3)
    plt.plot(n_list, block_conv_mses, 'bo-', label='Блочное')
    plt.plot(n_list, gaussian_conv_mses, 'ro-', label='Гауссовское')
    plt.xlabel('Размер ядра n')
    plt.ylabel('MSE')
    plt.title('Сравнение типов размытия (свёртка)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Сравнение методов для n=5
    plt.subplot(2, 2, 4)
    n = 5
    methods = ['Исходное', 'Блочное (свёртка)', 'Блочное (FFT)', 
               'Гауссовское (свёртка)', 'Гауссовское (FFT)']
    mses = [0, metrics[n]['block_conv_mse'], metrics[n]['block_fft_mse'],
            metrics[n]['gaussian_conv_mse'], metrics[n]['gaussian_fft_mse']]
    
    plt.bar(methods, mses)
    plt.ylabel('MSE')
    plt.title(f'Сравнение методов для n={n}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/task2/quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Детальный анализ спектров
    print("Детальный анализ спектров...")
    
    # Выбираем n=5 для детального анализа
    n = 5
    
    # Фурье-образы ядер
    h, w = original_image.shape
    k, l = n, n
    
    block_kernel_extended = np.zeros((h + k - 1, w + l - 1))
    gaussian_kernel_extended = np.zeros((h + k - 1, w + l - 1))
    
    start_h = (h + k - 1 - k) // 2
    start_w = (w + l - 1 - l) // 2
    
    block_kernel_extended[start_h:start_h+k, start_w:start_w+l] = block_kernels[n]
    gaussian_kernel_extended[start_h:start_h+k, start_w:start_w+l] = gaussian_kernels[n]
    
    fft_block_kernel = fftshift(fft2(block_kernel_extended))
    fft_gaussian_kernel = fftshift(fft2(gaussian_kernel_extended))
    
    # Визуализация спектров
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    fft_original_shifted = fftshift(fft2(original_image))
    log_magnitude_original = np.log(np.abs(fft_original_shifted) + 1)
    plt.imshow(log_magnitude_original, cmap='gray')
    plt.title('Спектр исходного изображения')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    log_magnitude_block = np.log(np.abs(fft_block_kernel) + 1)
    plt.imshow(log_magnitude_block, cmap='gray')
    plt.title('Спектр блочного ядра')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(gaussian_results[n], cmap='gray')
    plt.title('Гауссовское размытие')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    log_magnitude_gaussian = np.log(np.abs(fft_gaussian_kernel) + 1)
    plt.imshow(log_magnitude_gaussian, cmap='gray')
    plt.title('Спектр гауссовского ядра')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    # Сравнение спектров ядер
    center_y, center_x = log_magnitude_block.shape[0]//2, log_magnitude_block.shape[1]//2
    radius = min(center_y, center_x) // 4
    
    y, x = np.ogrid[:log_magnitude_block.shape[0], :log_magnitude_block.shape[1]]
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Радиальные профили
    angles = np.linspace(0, 2*np.pi, 360)
    radii = np.arange(0, radius, 2)
    
    block_radial = []
    gaussian_radial = []
    
    for r in radii:
        y_coords = center_y + r * np.cos(angles)
        x_coords = center_x + r * np.sin(angles)
        
        y_coords = np.clip(y_coords.astype(int), 0, log_magnitude_block.shape[0]-1)
        x_coords = np.clip(x_coords.astype(int), 0, log_magnitude_block.shape[1]-1)
        
        block_radial.append(np.mean(log_magnitude_block[y_coords, x_coords]))
        gaussian_radial.append(np.mean(log_magnitude_gaussian[y_coords, x_coords]))
    
    plt.plot(radii, block_radial, 'b-', label='Блочное ядро')
    plt.plot(radii, gaussian_radial, 'r-', label='Гауссовское ядро')
    plt.title('Радиальные профили спектров ядер')
    plt.xlabel('Радиус')
    plt.ylabel('Средняя амплитуда')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/task2/spectrum_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Вывод результатов
    print("\nРезультаты анализа размытия:")
    for n in n_values:
        print(f"\nРазмер ядра n={n}:")
        print(f"  Блочное размытие - MSE: {metrics[n]['block_conv_mse']:.6f}")
        print(f"  Гауссовское размытие - MSE: {metrics[n]['gaussian_conv_mse']:.6f}")
        print(f"  Разность методов (блочное): {metrics[n]['block_method_diff']:.6f}")
        print(f"  Разность методов (гауссовское): {metrics[n]['gaussian_method_diff']:.6f}")
    
    print("\nРазмытие изображений завершено!")
    print("Результаты сохранены в папке images/task2/")

if __name__ == "__main__":
    image_blurring() 