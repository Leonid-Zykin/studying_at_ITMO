import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import ndimage
from PIL import Image
import os

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task3', exist_ok=True)

def create_test_image():
    """Создает тестовое цветное изображение для увеличения резкости"""
    # Создаем изображение 256x256x3 (RGB)
    size = 256
    image = np.zeros((size, size, 3))
    
    # Красный канал - круг
    y, x = np.ogrid[:size, :size]
    circle = (x - size//4)**2 + (y - size//4)**2 <= (size//8)**2
    image[circle, 0] = 0.8  # Red channel
    
    # Зеленый канал - прямоугольник
    image[size//2-30:size//2+30, size//2-20:size//2+20, 1] = 0.6  # Green channel
    
    # Синий канал - диагональные линии
    for i in range(0, size, 20):
        image[i:i+3, :, 2] = 0.4  # Blue channel
    
    # Добавляем шум
    noise = np.random.normal(0, 0.05, (size, size, 3))
    image += noise
    
    # Ограничиваем значения
    image = np.clip(image, 0, 1)
    
    return image

def image_sharpening():
    """Увеличение резкости изображений"""
    
    # Загружаем готовое изображение
    print("Загрузка исходного изображения...")
    try:
        img_path = '../images/task3/original_image.png'
        img_pil = Image.open(img_path)
        original_image = np.array(img_pil) / 255.0
        print(f"Загружено изображение размером {original_image.shape}")
        
        # Проверяем, что изображение цветное
        if len(original_image.shape) == 2:
            print("Изображение черно-белое, конвертируем в RGB...")
            original_image = np.stack([original_image] * 3, axis=-1)
        elif original_image.shape[2] == 4:  # RGBA
            print("Изображение RGBA, конвертируем в RGB...")
            original_image = original_image[:, :, :3]
            
    except FileNotFoundError:
        print("Файл original_image.png не найден, создаем тестовое изображение...")
        original_image = create_test_image()
    
    # Получаем размеры изображения
    height, width, numberOfColorChannels = original_image.shape
    print(f"Размеры: {height}x{width}, каналов: {numberOfColorChannels}")
    
    # Сохраняем исходное изображение
    plt.figure(figsize=(12, 10))
    plt.imshow(original_image)
    plt.title('Исходное изображение', fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../images/task3/original_image.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Разделяем на цветовые каналы (как в MATLAB)
    IR = original_image[:, :, 0]  # Red channel
    IG = original_image[:, :, 1]  # Green channel  
    IB = original_image[:, :, 2]  # Blue channel
    
    # Создаем матрицу ядра увеличения резкости (точно как в MATLAB)
    sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]).reshape(3, 3)
    print("Ядро увеличения резкости:")
    print(sharp)
    
    # Сохраняем ядро
    plt.figure(figsize=(8, 6))
    plt.imshow(sharp, cmap='RdBu', vmin=-1, vmax=5)
    plt.title('Ядро увеличения резкости', fontsize=16, fontweight='bold')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../images/task3/sharpening_kernel.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Теперь будем сворачивать с изображением (как в MATLAB)
    print("Применение увеличения резкости с помощью свёртки...")
    
    # 1 раз
    IR_SHARP1 = ndimage.convolve(IR, sharp, mode='wrap')
    IG_SHARP1 = ndimage.convolve(IG, sharp, mode='wrap')
    IB_SHARP1 = ndimage.convolve(IB, sharp, mode='wrap')
    
    # 2 раза
    IR_SHARP2 = ndimage.convolve(IR_SHARP1, sharp, mode='wrap')
    IG_SHARP2 = ndimage.convolve(IG_SHARP1, sharp, mode='wrap')
    IB_SHARP2 = ndimage.convolve(IB_SHARP1, sharp, mode='wrap')
    
    # 3 раза
    IR_SHARP3 = ndimage.convolve(IR_SHARP2, sharp, mode='wrap')
    IG_SHARP3 = ndimage.convolve(IG_SHARP2, sharp, mode='wrap')
    IB_SHARP3 = ndimage.convolve(IB_SHARP2, sharp, mode='wrap')
    
    # Сохраняем результаты свёртки (как в MATLAB)
    I_SHARP1 = np.stack([IR_SHARP1, IG_SHARP1, IB_SHARP1], axis=-1)
    I_SHARP2 = np.stack([IR_SHARP2, IG_SHARP2, IB_SHARP2], axis=-1)
    I_SHARP3 = np.stack([IR_SHARP3, IG_SHARP3, IB_SHARP3], axis=-1)
    
    # Нормализуем и сохраняем
    for i, (sharp_img, name) in enumerate([(I_SHARP1, 'sharp1'), (I_SHARP2, 'sharp2'), (I_SHARP3, 'sharp3')]):
        # Нормализация в диапазон [0, 1]
        sharp_img_norm = np.clip(sharp_img, 0, 1)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(sharp_img_norm)
        plt.title(f'Увеличение резкости (свёртка) - {i+1} раз', fontsize=18, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'../images/task3/convolution_{name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Теперь переходим к анализу Фурье-образов (как в MATLAB)
    print("Применение увеличения резкости с помощью Фурье-преобразования...")
    
    # Фурье-образы исходного изображения
    IR_FFT = fft2(IR)
    IG_FFT = fft2(IG)
    IB_FFT = fft2(IB)
    
    # Фурье-образ ядра (с заполнением нулями до размера изображения)
    # Создаем ядро размером с изображение
    sharp_padded = np.zeros((height, width))
    sharp_padded[:3, :3] = sharp
    SHARP_FFT = fft2(sharp_padded)
    
    # Применяем теорему о свёртке (как в MATLAB)
    # 1 раз
    IR_SHARP1_FFT = IR_FFT * SHARP_FFT
    IG_SHARP1_FFT = IG_FFT * SHARP_FFT
    IB_SHARP1_FFT = IB_FFT * SHARP_FFT
    
    # 2 раза (ассоциативность)
    IR_SHARP2_FFT = IR_FFT * SHARP_FFT * SHARP_FFT
    IG_SHARP2_FFT = IG_FFT * SHARP_FFT * SHARP_FFT
    IB_SHARP2_FFT = IB_FFT * SHARP_FFT * SHARP_FFT
    
    # 3 раза
    IR_SHARP3_FFT = IR_FFT * SHARP_FFT * SHARP_FFT * SHARP_FFT
    IG_SHARP3_FFT = IG_FFT * SHARP_FFT * SHARP_FFT * SHARP_FFT
    IB_SHARP3_FFT = IB_FFT * SHARP_FFT * SHARP_FFT * SHARP_FFT
    
    # Обратное преобразование Фурье
    IR_SHARP1_FFT_result = np.real(ifft2(IR_SHARP1_FFT))
    IG_SHARP1_FFT_result = np.real(ifft2(IG_SHARP1_FFT))
    IB_SHARP1_FFT_result = np.real(ifft2(IB_SHARP1_FFT))
    
    IR_SHARP2_FFT_result = np.real(ifft2(IR_SHARP2_FFT))
    IG_SHARP2_FFT_result = np.real(ifft2(IG_SHARP2_FFT))
    IB_SHARP2_FFT_result = np.real(ifft2(IB_SHARP2_FFT))
    
    IR_SHARP3_FFT_result = np.real(ifft2(IR_SHARP3_FFT))
    IG_SHARP3_FFT_result = np.real(ifft2(IG_SHARP3_FFT))
    IB_SHARP3_FFT_result = np.real(ifft2(IB_SHARP3_FFT))
    
    # Сохраняем результаты FFT (как в MATLAB)
    I_SHARP1_FFT = np.stack([IR_SHARP1_FFT_result, IG_SHARP1_FFT_result, IB_SHARP1_FFT_result], axis=-1)
    I_SHARP2_FFT = np.stack([IR_SHARP2_FFT_result, IG_SHARP2_FFT_result, IB_SHARP2_FFT_result], axis=-1)
    I_SHARP3_FFT = np.stack([IR_SHARP3_FFT_result, IG_SHARP3_FFT_result, IB_SHARP3_FFT_result], axis=-1)
    
    # Нормализуем и сохраняем результаты FFT
    for i, (sharp_img, name) in enumerate([(I_SHARP1_FFT, 'sharp1_fft'), (I_SHARP2_FFT, 'sharp2_fft'), (I_SHARP3_FFT, 'sharp3_fft')]):
        # Нормализация в диапазон [0, 1]
        sharp_img_norm = np.clip(sharp_img, 0, 1)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(sharp_img_norm)
        plt.title(f'Увеличение резкости (FFT) - {i+1} раз', fontsize=18, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'../images/task3/fft_{name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Создаем сравнение методов для каждого количества применений
    print("Создание сравнения методов...")
    
    for i, (conv_img, fft_img, name) in enumerate([
        (I_SHARP1, I_SHARP1_FFT, 'sharp1'),
        (I_SHARP2, I_SHARP2_FFT, 'sharp2'),
        (I_SHARP3, I_SHARP3_FFT, 'sharp3')
    ]):
        plt.figure(figsize=(20, 15))
        
        # Исходное изображение
        plt.subplot(2, 3, 1)
        plt.imshow(original_image)
        plt.title('Исходное изображение', fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Свёртка
        plt.subplot(2, 3, 2)
        conv_norm = np.clip(conv_img, 0, 1)
        plt.imshow(conv_norm)
        plt.title(f'Свёртка - {i+1} раз', fontsize=16)
        plt.axis('off')
        
        # FFT
        plt.subplot(2, 3, 3)
        fft_norm = np.clip(fft_img, 0, 1)
        plt.imshow(fft_norm)
        plt.title(f'FFT - {i+1} раз', fontsize=16)
        plt.axis('off')
        
        # Разность методов
        diff = np.abs(conv_img - fft_img)
        max_diff = np.max(diff)
        
        plt.subplot(2, 3, 4)
        plt.imshow(diff, cmap='hot', vmin=0, vmax=max_diff)
        plt.title(f'Разность методов - {i+1} раз\nмакс: {max_diff:.6f}', fontsize=14)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # Детализация исходного
        plt.subplot(2, 3, 5)
        plt.imshow(original_image[height//4:3*height//4, width//4:3*width//4])
        plt.title('Детализация исходного', fontsize=14)
        plt.axis('off')
        
        # Детализация результата
        plt.subplot(2, 3, 6)
        plt.imshow(conv_norm[height//4:3*height//4, width//4:3*width//4])
        plt.title(f'Детализация результата - {i+1} раз', fontsize=14)
        plt.axis('off')
        
        plt.suptitle(f'Сравнение методов увеличения резкости - {i+1} применение', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(f'../images/task3/comparison_{name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Анализ качества увеличения резкости
    print("Анализ качества увеличения резкости...")
    
    # Вычисляем метрики качества
    metrics = {}
    
    for i, (conv_img, fft_img, name) in enumerate([
        (I_SHARP1, I_SHARP1_FFT, 'sharp1'),
        (I_SHARP2, I_SHARP2_FFT, 'sharp2'),
        (I_SHARP3, I_SHARP3_FFT, 'sharp3')
    ]):
        # MSE между исходным и результатом
        conv_mse = np.mean((original_image - conv_img)**2)
        fft_mse = np.mean((original_image - fft_img)**2)
        
        # Разность между методами
        method_diff = np.mean((conv_img - fft_img)**2)
        
        # Средняя яркость (показатель изменения контраста)
        conv_brightness = np.mean(conv_img)
        fft_brightness = np.mean(fft_img)
        original_brightness = np.mean(original_image)
        
        metrics[name] = {
            'conv_mse': conv_mse,
            'fft_mse': fft_mse,
            'method_diff': method_diff,
            'conv_brightness': conv_brightness,
            'fft_brightness': fft_brightness,
            'original_brightness': original_brightness
        }
    
    # График анализа качества
    plt.figure(figsize=(15, 10))
    
    # MSE для разных методов
    plt.subplot(2, 2, 1)
    names = ['1 раз', '2 раза', '3 раза']
    conv_mses = [metrics[f'sharp{i+1}']['conv_mse'] for i in range(3)]
    fft_mses = [metrics[f'sharp{i+1}']['fft_mse'] for i in range(3)]
    
    plt.plot(range(1, 4), conv_mses, 'bo-', label='Свёртка', linewidth=2, markersize=8)
    plt.plot(range(1, 4), fft_mses, 'rs--', label='FFT', linewidth=2, markersize=8)
    plt.xlabel('Количество применений', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Среднеквадратичная ошибка', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Разность между методами
    plt.subplot(2, 2, 2)
    method_diffs = [metrics[f'sharp{i+1}']['method_diff'] for i in range(3)]
    
    plt.plot(range(1, 4), method_diffs, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Количество применений', fontsize=12)
    plt.ylabel('Разность методов', fontsize=12)
    plt.title('Разность между свёрткой и FFT', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Изменение яркости
    plt.subplot(2, 2, 3)
    conv_brightnesses = [metrics[f'sharp{i+1}']['conv_brightness'] for i in range(3)]
    fft_brightnesses = [metrics[f'sharp{i+1}']['fft_brightness'] for i in range(3)]
    original_brightness = metrics['sharp1']['original_brightness']
    
    plt.axhline(y=original_brightness, color='k', linestyle='--', label='Исходная яркость')
    plt.plot(range(1, 4), conv_brightnesses, 'bo-', label='Свёртка', linewidth=2, markersize=8)
    plt.plot(range(1, 4), fft_brightnesses, 'rs--', label='FFT', linewidth=2, markersize=8)
    plt.xlabel('Количество применений', fontsize=12)
    plt.ylabel('Средняя яркость', fontsize=12)
    plt.title('Изменение яркости', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Сравнение для 2 применений
    plt.subplot(2, 2, 4)
    n = 2
    methods = ['Исходное', 'Свёртка\n(2 раза)', 'FFT\n(2 раза)']
    mses = [0, metrics[f'sharp{n}']['conv_mse'], metrics[f'sharp{n}']['fft_mse']]
    
    bars = plt.bar(methods, mses, color=['lightblue', 'blue', 'red'])
    plt.ylabel('MSE', fontsize=12)
    plt.title(f'Сравнение методов для {n} применений', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, mse in zip(bars, mses):
        if mse > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                    f'{mse:.6f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../images/task3/quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Вывод результатов
    print("\nРезультаты анализа увеличения резкости:")
    for i, name in enumerate(['sharp1', 'sharp2', 'sharp3']):
        print(f"\n{i+1} применение:")
        print(f"  Свёртка - MSE: {metrics[name]['conv_mse']:.6f}")
        print(f"  FFT - MSE: {metrics[name]['fft_mse']:.6f}")
        print(f"  Разность методов: {metrics[name]['method_diff']:.6f}")
        print(f"  Изменение яркости: {metrics[name]['conv_brightness']:.6f} (было {metrics[name]['original_brightness']:.6f})")
    
    print("\nУвеличение резкости завершено!")
    print("Результаты сохранены в папке images/task3/")
    print("\nСозданные изображения:")
    print("- original_image.png - исходное изображение")
    print("- sharpening_kernel.png - ядро увеличения резкости")
    for i in range(1, 4):
        print(f"- convolution_sharp{i}.png - увеличение резкости (свёртка) {i} раз")
        print(f"- fft_sharp{i}_fft.png - увеличение резкости (FFT) {i} раз")
        print(f"- comparison_sharp{i}.png - сравнение методов {i} раз")
    print("- quality_analysis.png - анализ качества")

if __name__ == "__main__":
    image_sharpening() 