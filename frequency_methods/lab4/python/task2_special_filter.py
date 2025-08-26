import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy import signal
import os

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task2', exist_ok=True)

# Параметры эксперимента
dt = 0.01
t = np.arange(-20, 20 + dt, dt)
N = len(t)

# Создаем прямоугольный импульс g(t)
a = 1.0
t1 = -2.0
t2 = 2.0
g = np.zeros_like(t)
mask = (t >= t1) & (t <= t2)
g[mask] = a

# Создаем зашумлённый сигнал
np.random.seed(42)
b = 0.0  # амплитуда случайного шума (нулевая для этого задания)
c = 0.5  # амплитуда гармонической помехи

# Исследуем различные частоты помехи
d_values = [5, 10, 15, 20]  # частоты помехи в Гц

# === Реализация фильтра (3): W2(p) = ((T1 p + 1)^2)/((T2 p + 1)(T3 p + 1)) ===

def create_W2_continuous(T1: float, T2: float, T3: float):
    """Возвращает коэффициенты непрерывной передаточной функции W2(s) = Num(s)/Den(s)."""
    # (T1 s + 1)^2 = T1^2 s^2 + 2 T1 s + 1
    num = np.array([T1**2, 2.0 * T1, 1.0], dtype=float)
    # (T2 s + 1)(T3 s + 1) = T2 T3 s^2 + (T2 + T3) s + 1
    den = np.array([T2 * T3, (T2 + T3), 1.0], dtype=float)
    return num, den

def discretize_bilinear(num_s, den_s, dt_seconds: float):
    """Дискретизация непрерывной системы по билинейному преобразованию (Tustin)."""
    sysd = signal.cont2discrete((num_s, den_s), dt_seconds, method='bilinear')
    b_z, a_z = sysd[0].flatten(), sysd[1].flatten()
    return b_z, a_z

def apply_iir(x: np.ndarray, b_z: np.ndarray, a_z: np.ndarray, use_filtfilt: bool = False) -> np.ndarray:
    """Применение IIR фильтра. Для иллюстрации формы можно использовать filtfilt (нуль-фазовый)."""
    if use_filtfilt:
        return signal.filtfilt(b_z, a_z, x, method='gust')
    # Для каузального отклика используем lfilter
    zi = signal.lfilter_zi(b_z, a_z) * x[0]
    y, _ = signal.lfilter(b_z, a_z, x, zi=zi)
    return y

# Подбор параметров фильтра под частоту помехи (эвристика)
# Идея: установить полосу пропускания ниже частоты d и усилить подавление нулями в числителе.
# T2, T3 определяют полюса (полосу), T1 — нули.

def select_T_params_for_noise(freq_hz: float):
    """Эвристический выбор T1, T2, T3 для подавления гармоники на freq_hz при сохранении НЧ прямоугольника."""
    # Цель: пропускать низкие частоты до ~2-3 Гц, подавлять в районе freq_hz.
    # Полюса (T2, T3) поставим на f_c_low ≈ 3 Гц: T ≈ 1/(2π f_c)
    f_c_low = 3.0
    T2 = 1.0 / (2.0 * np.pi * f_c_low)
    T3 = 1.0 / (2.0 * np.pi * f_c_low)
    # Нули (T1) дадут провал: выберем так, чтобы |W2| уменьшалось к freq_hz. Больше T1 — выше нули.
    # Свяжем T1 с целевой частотой подавления: ω = 2π f; грубо T1 ≈ k/ω
    omega = 2.0 * np.pi * freq_hz
    k = 0.5  # коэффициент подбора глубины провала
    T1 = k / max(omega, 1e-6)
    return T1, T2, T3

# График 1: Сравнение сигналов во временной области
plt.figure(figsize=(15, 10))

for i, d in enumerate(d_values):
    # Создаем зашумлённый сигнал с гармонической помехой
    harmonic_noise = c * np.sin(2 * np.pi * d * t)
    u = g + harmonic_noise

    # Подбираем параметры фильтра по (3)
    T1, T2, T3 = select_T_params_for_noise(d)
    num_s, den_s = create_W2_continuous(T1, T2, T3)
    b_z, a_z = discretize_bilinear(num_s, den_s, dt)

    # Применяем фильтр: показываем честный каузальный и нуль-фазовый для сравнения
    u_filtered = apply_iir(u, b_z, a_z, use_filtfilt=False)
    u_filtered_nf = apply_iir(u, b_z, a_z, use_filtfilt=True)

    plt.subplot(2, 2, i+1)
    mask_plot = (t >= t1-1) & (t <= t2+1)

    plt.plot(t[mask_plot], g[mask_plot], 'g-', linewidth=2, label='Исходный сигнал g(t)')
    plt.plot(t[mask_plot], u[mask_plot], 'r-', alpha=0.7, linewidth=1, label=f'С помехой (d={d} Гц)')
    plt.plot(t[mask_plot], u_filtered[mask_plot], 'b-', linewidth=2, label='Отфильтрованный (lfilter)')
    # plt.plot(t[mask_plot], u_filtered_nf[mask_plot], 'c--', linewidth=2, label='Отфильтрованный (filtfilt)')

    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.title(f'Специальный фильтр (3), d = {d} Гц')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task2/special_filter_time_domain.png', dpi=300, bbox_inches='tight')
plt.close()

# График 2: Фурье-образы
plt.figure(figsize=(15, 10))

for i, d in enumerate(d_values):
    harmonic_noise = c * np.sin(2 * np.pi * d * t)
    u = g + harmonic_noise

    T1, T2, T3 = select_T_params_for_noise(d)
    num_s, den_s = create_W2_continuous(T1, T2, T3)
    b_z, a_z = discretize_bilinear(num_s, den_s, dt)

    u_filtered = apply_iir(u, b_z, a_z, use_filtfilt=False)

    # Вычисляем Фурье-образы
    U = fftshift(fft(u))
    U_filtered = fftshift(fft(u_filtered))

    # Создаем массив частот
    T_total = t[-1] - t[0]
    df = 1 / T_total
    f = np.linspace(-N//2, N//2, N, endpoint=False) * df

    plt.subplot(2, 2, i+1)

    # Показываем частоты до 30 Гц
    mask_freq = (f >= 0) & (f <= 30)

    plt.plot(f[mask_freq], np.abs(U[mask_freq]), 'r-', alpha=0.7, linewidth=1, label='Исходный сигнал')
    plt.plot(f[mask_freq], np.abs(U_filtered[mask_freq]), 'g-', linewidth=2, label='Отфильтрованный')
    plt.axvline(x=d, color='orange', linestyle='--', alpha=0.7, label=f'Частота помехи {d} Гц')

    plt.xlabel('Частота f')
    plt.ylabel('|U(f)|')
    plt.title(f'Фурье-образы, d = {d} Гц')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task2/special_filter_freq_domain.png', dpi=300, bbox_inches='tight')
plt.close()

# График 3: АЧХ фактического дискретного фильтра
plt.figure(figsize=(15, 10))

for i, d in enumerate(d_values):
    T1, T2, T3 = select_T_params_for_noise(d)
    num_s, den_s = create_W2_continuous(T1, T2, T3)
    b_z, a_z = discretize_bilinear(num_s, den_s, dt)

    # Частотная характеристика дискретного фильтра
    w, h = signal.freqz(b_z, a_z, worN=2048, fs=1.0/dt)

    plt.subplot(2, 2, i+1)
    plt.semilogx(w, np.abs(h), 'b-', linewidth=2, label='|W2(e^{jω})| (discrete)')
    plt.axvline(x=d, color='red', linestyle='--', alpha=0.7, label=f'Частота помехи {d} Гц')

    plt.xlabel('Частота f (Гц)')
    plt.ylabel('|W|')
    plt.title(f'АЧХ фильтра (3), d = {d} Гц')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.xlim(0.1, 30)  # До 30 Гц для наглядности
    plt.ylim(0, 1.1)

plt.tight_layout()
plt.savefig('../images/task2/special_filter_frequency_response.png', dpi=300, bbox_inches='tight')
plt.close()

# Анализ влияния параметра c на эффективность фильтрации
plt.figure(figsize=(15, 10))

d_fixed = 10  # фиксированная частота помехи
c_values = [0.2, 0.5, 0.8, 1.2]

for i, c_val in enumerate(c_values):
    harmonic_noise = c_val * np.sin(2 * np.pi * d_fixed * t)
    u_c = g + harmonic_noise

    T1, T2, T3 = select_T_params_for_noise(d_fixed)
    num_s, den_s = create_W2_continuous(T1, T2, T3)
    b_z, a_z = discretize_bilinear(num_s, den_s, dt)
    u_filtered_c = apply_iir(u_c, b_z, a_z, use_filtfilt=False)

    plt.subplot(2, 2, i+1)
    mask_plot = (t >= t1-1) & (t <= t2+1)

    plt.plot(t[mask_plot], g[mask_plot], 'g-', linewidth=2, label='Исходный сигнал')
    plt.plot(t[mask_plot], u_c[mask_plot], 'r-', alpha=0.7, linewidth=1, label=f'С помехой (c={c_val})')
    plt.plot(t[mask_plot], u_filtered_c[mask_plot], 'b-', linewidth=2, label='Отфильтрованный')

    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.title(f'Влияние параметра c, c = {c_val}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task2/special_filter_c_influence.png', dpi=300, bbox_inches='tight')
plt.close()

# Количественный анализ эффективности фильтрации
print("Анализ специального фильтра (формула 3):")
print("=" * 50)

for d in d_values:
    harmonic_noise = c * np.sin(2 * np.pi * d * t)
    u = g + harmonic_noise

    T1, T2, T3 = select_T_params_for_noise(d)
    num_s, den_s = create_W2_continuous(T1, T2, T3)
    b_z, a_z = discretize_bilinear(num_s, den_s, dt)
    u_filtered = apply_iir(u, b_z, a_z, use_filtfilt=False)

    # Вычисляем среднеквадратичную ошибку
    mse = np.mean((g - u_filtered)**2)
    # Вычисляем корреляцию с исходным сигналом
    correlation = np.corrcoef(g, u_filtered)[0, 1]

    # Подавление помехи на частоте d
    noise_power_original = np.mean(harmonic_noise**2)
    noise_power_filtered = np.mean((u_filtered - g)**2)
    suppression = 10 * np.log10(noise_power_original / noise_power_filtered) if noise_power_filtered > 0 else float('inf')

    print(f"d = {d} Гц:")
    print(f"  Параметры фильтра (3): T1={T1:.4f}, T2={T2:.4f}, T3={T3:.4f}")
    print(f"  Среднеквадратичная ошибка: {mse:.6f}")
    print(f"  Корреляция с исходным сигналом: {correlation:.6f}")
    print(f"  Подавление помехи: {suppression:.2f} дБ")
    print()

print(f"Параметры: a={a}, t1={t1}, t2={t2}, b={b}, c={c}")
print(f"Исследованные значения d: {d_values}")