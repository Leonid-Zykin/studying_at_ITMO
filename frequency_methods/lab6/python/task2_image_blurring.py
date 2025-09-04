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
    
    # Загружаем готовое изображение
    print("Загрузка исходного изображения...")
    try:
        from PIL import Image
        img_path = '../images/task2/original_image.png'
        img_pil = Image.open(img_path).convert('L')
        original_image = np.array(img_pil) / 255.0
        print(f"Загружено изображение размером {original_image.shape}")
    except FileNotFoundError:
        print("Файл original_image.png не найден, создаем тестовое изображение...")
        original_image = create_test_image()
    
    # Сохраняем исходное изображение
    plt.figure(figsize=(12, 10))
    plt.imshow(original_image, cmap='gray')
    plt.title('Исходное изображение', fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../images/task2/original_image.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Упрощаем: только 2 значения n для лучшей видимости
    n_values = [5, 9]  # Убрал n=3, оставил только 5 и 9
    
    # Создаем ядра размытия
    print("Создание ядер размытия...")
    block_kernels = {}
    gaussian_kernels = {}
    
    for n in n_values:
        block_kernels[n] = create_block_kernel(n)
        gaussian_kernels[n] = create_gaussian_kernel(n)
    
    # Визуализация ядер - каждое на отдельной картинке
    for n in n_values:
        # Блочное ядро
        plt.figure(figsize=(10, 8))
        plt.imshow(block_kernels[n], cmap='gray')
        plt.title(f'Блочное ядро размытия n={n}', fontsize=16, fontweight='bold')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'../images/task2/block_kernel_n{n}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Гауссовское ядро
        plt.figure(figsize=(10, 8))
        plt.imshow(gaussian_kernels[n], cmap='gray')
        plt.title(f'Гауссовское ядро размытия n={n}', fontsize=16, fontweight='bold')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
    plt.tight_layout()
        plt.savefig(f'../images/task2/gaussian_kernel_n{n}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Применяем размытие с помощью свёртки
    print("Применение размытия с помощью свёртки...")
    block_results = {}
    gaussian_results = {}
    
    for n in n_values:
        # Блочное размытие
        block_results[n] = ndimage.convolve(original_image, block_kernels[n], mode='wrap')
        
        # Гауссовское размытие
        gaussian_results[n] = ndimage.convolve(original_image, gaussian_kernels[n], mode='wrap')
    
    # Визуализация результатов - КАЖДОЕ НА ОТДЕЛЬНОЙ КАРТИНКЕ
    for n in n_values:
        # Блочное размытие
        plt.figure(figsize=(12, 10))
        plt.imshow(block_results[n], cmap='gray')
        plt.title(f'Блочное размытие n={n}', fontsize=18, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'../images/task2/block_blur_n{n}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Гауссовское размытие
        plt.figure(figsize=(12, 10))
        plt.imshow(gaussian_results[n], cmap='gray')
        plt.title(f'Гауссовское размытие n={n}', fontsize=18, fontweight='bold')
        plt.axis('off')
    plt.tight_layout()
        plt.savefig(f'../images/task2/gaussian_blur_n{n}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Применяем размытие с помощью Фурье-преобразования
    print("Применение размытия с помощью Фурье-преобразования...")
    block_fft_results = {}
    gaussian_fft_results = {}
    
    # Фурье-образ исходного изображения
    fft_original = fft2(original_image)
    
    for n in n_values:
        # Создаем ядра того же размера, что и изображение (для циклической свертки)
        h, w = original_image.shape
        
        # Создаем ядра размером с изображение
        block_kernel_full = np.zeros((h, w))
        gaussian_kernel_full = np.zeros((h, w))
        
        # Размещаем ядра в левом верхнем углу (для циклической свертки)
        block_kernel_full[:n, :n] = block_kernels[n]
        gaussian_kernel_full[:n, :n] = gaussian_kernels[n]
        
        # Фурье-образы ядер
        fft_block_kernel = fft2(block_kernel_full)
        fft_gaussian_kernel = fft2(gaussian_kernel_full)
        
        # Поэлементное умножение в частотной области (циклическая свертка)
        fft_block_result = fft_original * fft_block_kernel
        fft_gaussian_result = fft_original * fft_gaussian_kernel
        
        # Обратное преобразование
        block_fft_results[n] = np.real(ifft2(fft_block_result))
        gaussian_fft_results[n] = np.real(ifft2(fft_gaussian_result))
    
    # Визуализация результатов Фурье-метода - КАЖДОЕ НА ОТДЕЛЬНОЙ КАРТИНКЕ
    for n in n_values:
        # Блочное размытие FFT
        plt.figure(figsize=(12, 10))
        plt.imshow(block_fft_results[n], cmap='gray')
        plt.title(f'Блочное размытие (FFT) n={n}', fontsize=18, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'../images/task2/block_blur_fft_n{n}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Гауссовское размытие FFT
        plt.figure(figsize=(12, 10))
        plt.imshow(gaussian_fft_results[n], cmap='gray')
        plt.title(f'Гауссовское размытие (FFT) n={n}', fontsize=18, fontweight='bold')
        plt.axis('off')
    plt.tight_layout()
        plt.savefig(f'../images/task2/gaussian_blur_fft_n{n}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Создаем ОДНУ общую картинку сравнения методов для каждого n
    for n in n_values:
    plt.figure(figsize=(20, 15))
    
    # Исходное изображение
        plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=1)
    plt.title('Исходное изображение', fontsize=16, fontweight='bold')
    plt.axis('off')
    
        # Блочное размытие - свертка
        plt.subplot(2, 3, 2)
        plt.imshow(block_results[n], cmap='gray', vmin=0, vmax=1)
        plt.title(f'Блочное размытие (свёртка) n={n}', fontsize=16)
        plt.axis('off')
        
        # Блочное размытие - FFT
        plt.subplot(2, 3, 3)
        plt.imshow(block_fft_results[n], cmap='gray', vmin=0, vmax=1)
        plt.title(f'Блочное размытие (FFT) n={n}', fontsize=16)
        plt.axis('off')
        
        # Гауссовское размытие - свертка
        plt.subplot(2, 3, 4)
        plt.imshow(gaussian_results[n], cmap='gray', vmin=0, vmax=1)
        plt.title(f'Гауссовское размытие (свёртка) n={n}', fontsize=16)
        plt.axis('off')
        
        # Гауссовское размытие - FFT
        plt.subplot(2, 3, 5)
        plt.imshow(gaussian_fft_results[n], cmap='gray', vmin=0, vmax=1)
        plt.title(f'Гауссовское размытие (FFT) n={n}', fontsize=16)
        plt.axis('off')
        
        # Разность между методами (свёртка vs FFT) для блочного
        diff_block = np.abs(block_results[n] - block_fft_results[n])
        max_diff_block = np.max(diff_block)
        plt.subplot(2, 3, 6)
        im_diff = plt.imshow(diff_block, cmap='hot', vmin=0, vmax=max_diff_block)
        plt.title(f'Разность методов (блочное)\nn={n}, макс: {max_diff_block:.6f}', fontsize=14)
        plt.colorbar(im_diff, fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.suptitle(f'Сравнение методов размытия для n={n}', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(f'../images/task2/comparison_n{n}.png', dpi=300, bbox_inches='tight')
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
        gaussian_fft_mse = np.mean((original_image - gaussian_results[n])**2)
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
    
    plt.plot(n_list, block_conv_mses, 'bo-', label='Блочное (свёртка)', linewidth=2, markersize=8)
    plt.plot(n_list, block_fft_mses, 'bs--', label='Блочное (FFT)', linewidth=2, markersize=8)
    plt.plot(n_list, gaussian_conv_mses, 'ro-', label='Гауссовское (свёртка)', linewidth=2, markersize=8)
    plt.plot(n_list, gaussian_fft_mses, 'rs--', label='Гауссовское (FFT)', linewidth=2, markersize=8)
    plt.xlabel('Размер ядра n', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Среднеквадратичная ошибка', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Разность между методами
    plt.subplot(2, 2, 2)
    block_method_diffs = [metrics[n]['block_method_diff'] for n in n_list]
    gaussian_method_diffs = [metrics[n]['gaussian_method_diff'] for n in n_list]
    
    plt.plot(n_list, block_method_diffs, 'bo-', label='Блочное размытие', linewidth=2, markersize=8)
    plt.plot(n_list, gaussian_method_diffs, 'ro-', label='Гауссовское размытие', linewidth=2, markersize=8)
    plt.xlabel('Размер ядра n', fontsize=12)
    plt.ylabel('Разность методов', fontsize=12)
    plt.title('Разность между свёрткой и FFT', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Сравнение типов размытия
    plt.subplot(2, 2, 3)
    plt.plot(n_list, block_conv_mses, 'bo-', label='Блочное', linewidth=2, markersize=8)
    plt.plot(n_list, gaussian_conv_mses, 'ro-', label='Гауссовское', linewidth=2, markersize=8)
    plt.xlabel('Размер ядра n', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Сравнение типов размытия (свёртка)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Сравнение методов для n=5
    plt.subplot(2, 2, 4)
    n = 5
    methods = ['Исходное', 'Блочное\n(свёртка)', 'Блочное\n(FFT)', 
               'Гауссовское\n(свёртка)', 'Гауссовское\n(FFT)']
    mses = [0, metrics[n]['block_conv_mse'], metrics[n]['block_fft_mse'],
            metrics[n]['gaussian_conv_mse'], metrics[n]['gaussian_fft_mse']]
    
    bars = plt.bar(methods, mses, color=['lightblue', 'blue', 'lightblue', 'red', 'lightcoral'])
    plt.ylabel('MSE', fontsize=12)
    plt.title(f'Сравнение методов для n={n}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, mse in zip(bars, mses):
        if mse > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                    f'{mse:.6f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../images/task2/quality_analysis.png', dpi=300, bbox_inches='tight')
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
    print("\nСозданные изображения:")
    print("- original_image.png - исходное изображение")
    for n in n_values:
        print(f"- block_kernel_n{n}.png - блочное ядро n={n}")
        print(f"- gaussian_kernel_n{n}.png - гауссовское ядро n={n}")
        print(f"- block_blur_n{n}.png - блочное размытие n={n}")
        print(f"- gaussian_blur_n{n}.png - гауссовское размытие n={n}")
        print(f"- block_blur_fft_n{n}.png - блочное размытие FFT n={n}")
        print(f"- gaussian_blur_fft_n{n}.png - гауссовское размытие FFT n={n}")
        print(f"- comparison_n{n}.png - сравнение методов для n={n}")
    print("- quality_analysis.png - анализ качества")

if __name__ == "__main__":
    image_blurring() 