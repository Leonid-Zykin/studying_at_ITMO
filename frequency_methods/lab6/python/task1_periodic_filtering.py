import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
import os

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task1', exist_ok=True)

def task1_periodic_filtering():
    """Задание 1. Фильтрация изображений с периодичностью."""
    
    print("Загрузка исходного изображения...")
    
    # Загружаем исходное изображение (только для чтения!)
    img_path = '../images/task1/original_image.png'
    try:
        img_pil = Image.open(img_path).convert('L')
        I_norm = np.array(img_pil, dtype=np.float64) / 255.0
        print(f"Загружено изображение размером {I_norm.shape}")
    except FileNotFoundError:
        print(f"Файл {img_path} не найден!")
        return
    
    # 1. Прямое Фурье-преобразование
    print("Выполнение прямого Фурье-преобразования...")
    I_FFT = fftshift(fft2(I_norm))
    I_FFT_abs = np.abs(I_FFT)
    I_FFT_angle = np.angle(I_FFT)
    I_FFT_abs_log = np.log(1 + I_FFT_abs)
    
    # Нормализация логарифмированного спектра
    I_FFT_abs_log_norm = I_FFT_abs_log - np.min(I_FFT_abs_log)
    I_FFT_abs_log_norm = I_FFT_abs_log_norm / np.max(I_FFT_abs_log_norm)
    
    # Сохраняем спектр для ручного редактирования в PAINT
    # Это изображение нужно будет отредактировать вручную
    spectrum_img = Image.fromarray((I_FFT_abs_log_norm * 255).astype(np.uint8))
    spectrum_img.save('../images/task1/FFT_IMAGE.png')
    print("Спектр сохранен в FFT_IMAGE.png для ручного редактирования")
    
    # 2. Визуализация исходного изображения и его спектра
    plt.figure(figsize=(15, 10))
    
    # Исходное изображение
    plt.subplot(2, 3, 1)
    plt.imshow(I_norm, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    # Логарифмированный спектр
    plt.subplot(2, 3, 2)
    plt.imshow(I_FFT_abs_log_norm, cmap='gray')
    plt.title('Логарифмированный спектр')
    plt.axis('off')
    
    # Спектр в линейном масштабе
    plt.subplot(2, 3, 3)
    plt.imshow(I_FFT_abs_log, cmap='gray')
    plt.title('Спектр (линейный масштаб)')
    plt.axis('off')
    
    # Фаза спектра
    plt.subplot(2, 3, 4)
    plt.imshow(I_FFT_angle, cmap='gray')
    plt.title('Фаза спектра')
    plt.axis('off')
    
    # Детали спектра (центр)
    center_y, center_x = I_FFT_abs_log_norm.shape[0]//2, I_FFT_abs_log_norm.shape[1]//2
    crop_size = 100
    crop_y = slice(center_y-crop_size//2, center_y+crop_size//2)
    crop_x = slice(center_x-crop_size//2, center_x+crop_size//2)
    
    plt.subplot(2, 3, 5)
    plt.imshow(I_FFT_abs_log_norm[crop_y, crop_x], cmap='gray')
    plt.title('Деталь спектра (центр)')
    plt.axis('off')
    
    # Радиальный профиль спектра
    plt.subplot(2, 3, 6)
    y, x = np.ogrid[:I_FFT_abs_log_norm.shape[0], :I_FFT_abs_log_norm.shape[1]]
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Создаем радиальные кольца
    max_radius = min(center_x, center_y)
    radii = np.arange(0, max_radius, 5)
    radial_profile = []
    
    for r in radii:
        mask = (distance_from_center >= r) & (distance_from_center < r + 5)
        if np.any(mask):
            radial_profile.append(np.mean(I_FFT_abs_log_norm[mask]))
        else:
            radial_profile.append(0)
    
    plt.plot(radii, radial_profile, 'b-')
    plt.title('Радиальный профиль спектра')
    plt.xlabel('Радиус')
    plt.ylabel('Средняя амплитуда')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/task1/fourier_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Симуляция ручного редактирования (заглушка)
    # В реальности это изображение редактируется в PAINT
    print("Симуляция ручного редактирования спектра...")
    
    # Создаем "отредактированный" спектр (убираем яркие пики)
    I_FFT_abs_log_norm_edited = I_FFT_abs_log_norm.copy()
    
    # Находим яркие пики (периодические компоненты)
    threshold = np.percentile(I_FFT_abs_log_norm, 94)  # Оптимальный порог
    bright_pixels = I_FFT_abs_log_norm > threshold
    
    # Создаем маску для подавления периодичности
    # В реальности это делается вручную в PAINT
    mask = np.ones_like(I_FFT_abs_log_norm)
    
    # Подавляем яркие пики (симуляция ручного редактирования)
    for i in range(I_FFT_abs_log_norm.shape[0]):
        for j in range(I_FFT_abs_log_norm.shape[1]):
            if bright_pixels[i, j]:
                # Создаем плавное затухание вокруг пика
                distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                if distance > 20:  # Не трогаем центр (DC компонент)
                    # Плавное затухание
                    decay = np.exp(-((i - center_y)**2 + (j - center_x)**2) / (2 * 50**2))
                    mask[i, j] = 0.3 + 0.7 * decay
    
    # Применяем маску
    I_FFT_abs_log_norm_edited = I_FFT_abs_log_norm_edited * mask
    
    # Сохраняем "отредактированный" спектр
    edited_spectrum_img = Image.fromarray((I_FFT_abs_log_norm_edited * 255).astype(np.uint8))
    edited_spectrum_img.save('../images/task1/FFT_IMAGE_edited.png')
    
    # 4. Обратное преобразование
    print("Выполнение обратного Фурье-преобразования...")
    
    # Обратное логарифмирование и нормализация
    I_FFT_abs_log_edited = I_FFT_abs_log_norm_edited * (np.max(I_FFT_abs_log) - np.min(I_FFT_abs_log)) + np.min(I_FFT_abs_log)
    I_FFT_abs_edited = np.exp(I_FFT_abs_log_edited) - 1
    
    # Восстановление Фурье-образа
    I_FFT_edited = I_FFT_abs_edited * np.exp(1j * I_FFT_angle)
    
    # Обратное Фурье-преобразование
    I_IFT_edited = ifft2(ifftshift(I_FFT_edited))
    
    # Нормализация восстановленного изображения
    I_IFT_edited_norm = np.real(I_IFT_edited)
    I_IFT_edited_norm = I_IFT_edited_norm - np.min(I_IFT_edited_norm)
    I_IFT_edited_norm = I_IFT_edited_norm / np.max(I_IFT_edited_norm)
    
    # Сохраняем восстановленное изображение
    restored_img = Image.fromarray((I_IFT_edited_norm * 255).astype(np.uint8))
    restored_img.save('../images/task1/ship_restored2.png')
    
    # 5. Сравнение результатов
    plt.figure(figsize=(18, 12))
    
    # Исходное изображение
    plt.subplot(2, 4, 1)
    plt.imshow(I_norm, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    # Спектр исходного
    plt.subplot(2, 4, 2)
    plt.imshow(I_FFT_abs_log_norm, cmap='gray')
    plt.title('Спектр исходного')
    plt.axis('off')
    
    # Отредактированный спектр
    plt.subplot(2, 4, 3)
    plt.imshow(I_FFT_abs_log_norm_edited, cmap='gray')
    plt.title('Отредактированный спектр')
    plt.axis('off')
    
    # Восстановленное изображение
    plt.subplot(2, 4, 4)
    plt.imshow(I_IFT_edited_norm, cmap='gray')
    plt.title('Восстановленное изображение')
    plt.axis('off')
    
    # Детали исходного
    crop_size = 80
    crop_y = slice(center_y-crop_size//2, center_y+crop_size//2)
    crop_x = slice(center_x-crop_size//2, center_x+crop_size//2)
    
    plt.subplot(2, 4, 5)
    plt.imshow(I_norm[crop_y, crop_x], cmap='gray')
    plt.title('Деталь исходного')
    plt.axis('off')
    
    # Детали восстановленного
    plt.subplot(2, 4, 6)
    plt.imshow(I_IFT_edited_norm[crop_y, crop_x], cmap='gray')
    plt.title('Деталь восстановленного')
    plt.axis('off')
    
    # Разность (подавленная периодичность)
    difference = np.abs(I_norm - I_IFT_edited_norm)
    plt.subplot(2, 4, 7)
    plt.imshow(difference, cmap='hot')
    plt.title('Разность (подавленная периодичность)')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Профиль подавления
    plt.subplot(2, 4, 8)
    # Горизонтальный срез через центр
    center_slice = difference[center_y, :]
    plt.plot(center_slice, 'b-', linewidth=2)
    plt.title('Профиль подавления (горизонталь)')
    plt.xlabel('Пиксели')
    plt.ylabel('Разность')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/task1/filtering_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Анализ качества фильтрации
    print("Анализ качества фильтрации...")
    
    # Вычисляем метрики
    original_variance = np.var(I_norm)
    restored_variance = np.var(I_IFT_edited_norm)
    suppression_ratio = original_variance / restored_variance
    
    # Анализ периодичности через автокорреляцию
    def autocorr2d(img):
        """2D автокорреляция для оценки периодичности"""
        fft_img = fft2(img)
        autocorr = np.real(ifft2(fft_img * np.conj(fft_img)))
        return fftshift(autocorr)
    
    autocorr_original = autocorr2d(I_norm)
    autocorr_restored = autocorr2d(I_IFT_edited_norm)
    
    # Нормализация автокорреляции
    autocorr_original = autocorr_original / np.max(autocorr_original)
    autocorr_restored = autocorr_restored / np.max(autocorr_restored)
    
    # Визуализация автокорреляции
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 3, 1)
    plt.imshow(I_norm, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(autocorr_original, cmap='hot')
    plt.title('Автокорреляция исходного')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(autocorr_restored, cmap='hot')
    plt.title('Автокорреляция восстановленного')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/task1/autocorrelation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Вывод результатов
    print("\nРезультаты фильтрации периодичности:")
    print(f"  Дисперсия исходного изображения: {original_variance:.6f}")
    print(f"  Дисперсия восстановленного: {restored_variance:.6f}")
    print(f"  Коэффициент подавления: {suppression_ratio:.3f}")
    print(f"  Подавление периодичности: {(1 - 1/suppression_ratio)*100:.1f}%")
    
    print("\nФильтрация периодичности завершена!")
    print("Результаты сохранены в папке images/task1/")
    print("\nПримечание: FFT_IMAGE.png нужно отредактировать вручную в PAINT")
    print("для удаления ярких пиков (периодических компонентов)")

if __name__ == "__main__":
    task1_periodic_filtering() 