import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import os
from datetime import datetime, timedelta

# Отключаем интерактивный режим matplotlib
plt.ioff()

# Создаем папку для сохранения изображений
os.makedirs('../images/task3', exist_ok=True)

# Создаем синтетические данные о стоимости акций Сбербанка
# В реальном случае здесь был бы код для загрузки данных из файла
def create_stock_data():
    """Создает синтетические данные о стоимости акций"""
    
    # Создаем временной ряд на 2 года
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Базовая стоимость акции
    base_price = 250.0
    
    # Тренд (рост)
    trend = np.linspace(0, 50, len(dates))
    
    # Сезонность (годовые циклы)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    
    # Случайные колебания
    np.random.seed(42)
    noise = np.random.normal(0, 5, len(dates))
    
    # Долгосрочные тренды
    long_trend = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / (365 * 2))
    
    # Итоговая цена
    prices = base_price + trend + seasonality + noise + long_trend
    
    # Создаем DataFrame
    stock_data = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    
    return stock_data

# Загружаем данные
stock_data = create_stock_data()
dates = stock_data['Date'].values
prices = stock_data['Close'].values

# Функция для создания фильтра первого порядка
def create_first_order_filter(T):
    """Создает фильтр первого порядка с передаточной функцией W(p) = 1/(T*p + 1)"""
    # В дискретном времени: W(z) = 1/(T*(z-1)/(dt*z) + 1)
    # Приводим к виду: W(z) = (dt*z)/(T*(z-1) + dt*z) = (dt*z)/((T+dt)*z - T)
    # Нормализуем: W(z) = (dt/(T+dt)*z)/(z - T/(T+dt))
    
    dt = 1  # один день
    alpha = dt / (T + dt)
    beta = T / (T + dt)
    
    # Коэффициенты фильтра: y[n] = alpha * x[n] + beta * y[n-1]
    return alpha, beta

# Функция для применения фильтра с исправлением начального значения
def apply_filter_with_correction(x, alpha, beta):
    """Применяет фильтр первого порядка к сигналу x с исправлением начального значения"""
    y = np.zeros_like(x)
    y[0] = x[0]  # начальное значение равно первому значению входного сигнала
    
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + beta * y[i-1]
    
    return y

# Исследуем различные постоянные времени
T_values = {
    '1 день': 1,
    '1 неделя': 7,
    '1 месяц': 30,
    '3 месяца': 90,
    '1 год': 365
}

# График 1: Сравнение исходного и отфильтрованных сигналов
plt.figure(figsize=(20, 12))

# Исходный сигнал
plt.subplot(2, 1, 1)
plt.plot(dates, prices, 'b-', linewidth=1, alpha=0.8, label='Исходные данные')
plt.xlabel('Дата')
plt.ylabel('Стоимость акции (руб.)')
plt.title('Исходные данные о стоимости акций Сбербанка')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Отфильтрованные сигналы
plt.subplot(2, 1, 2)
colors = ['red', 'orange', 'green', 'purple', 'brown']
for i, (period, T) in enumerate(T_values.items()):
    alpha, beta = create_first_order_filter(T)
    filtered_prices = apply_filter_with_correction(prices, alpha, beta)
    plt.plot(dates, filtered_prices, color=colors[i], linewidth=2, label=f'Отфильтрованный ({period})')

plt.xlabel('Дата')
plt.ylabel('Стоимость акции (руб.)')
plt.title('Сглаженные данные с различными постоянными времени')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('../images/task3/stock_data_smoothing_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# График 2: Детальное сравнение для каждого периода
fig, axes = plt.subplots(3, 2, figsize=(20, 15))
axes = axes.flatten()

for i, (period, T) in enumerate(T_values.items()):
    alpha, beta = create_first_order_filter(T)
    filtered_prices = apply_filter_with_correction(prices, alpha, beta)
    
    # Показываем последние 6 месяцев для детального анализа
    mask_recent = dates >= dates[-180]
    
    axes[i].plot(dates[mask_recent], prices[mask_recent], 'b-', linewidth=1, alpha=0.7, label='Исходные данные')
    axes[i].plot(dates[mask_recent], filtered_prices[mask_recent], 'r-', linewidth=2, label=f'Отфильтрованный ({period})')
    
    axes[i].set_xlabel('Дата')
    axes[i].set_ylabel('Стоимость акции (руб.)')
    axes[i].set_title(f'Сглаживание: {period}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    axes[i].tick_params(axis='x', rotation=45)

# Убираем лишний подграфик
axes[-1].set_visible(False)

plt.tight_layout()
plt.savefig('../images/task3/stock_data_smoothing_detailed.png', dpi=300, bbox_inches='tight')
plt.close()

# График 3: Анализ эффективности сглаживания
plt.figure(figsize=(15, 10))

# Статистика сглаживания
periods = list(T_values.keys())
mse_values = []
correlation_values = []
smoothing_factors = []

for period, T in T_values.items():
    alpha, beta = create_first_order_filter(T)
    filtered_prices = apply_filter_with_correction(prices, alpha, beta)
    
    # Среднеквадратичная ошибка
    mse = np.mean((prices - filtered_prices)**2)
    mse_values.append(mse)
    
    # Корреляция с исходным сигналом
    correlation = np.corrcoef(prices, filtered_prices)[0, 1]
    correlation_values.append(correlation)
    
    # Фактор сглаживания (отношение стандартных отклонений)
    smoothing_factor = np.std(filtered_prices) / np.std(prices)
    smoothing_factors.append(smoothing_factor)

# График среднеквадратичной ошибки
plt.subplot(2, 2, 1)
plt.bar(periods, mse_values, color='skyblue')
plt.xlabel('Период фильтрации')
plt.ylabel('Среднеквадратичная ошибка')
plt.title('Среднеквадратичная ошибка фильтрации')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# График корреляции
plt.subplot(2, 2, 2)
plt.bar(periods, correlation_values, color='lightcoral')
plt.xlabel('Период фильтрации')
plt.ylabel('Корреляция')
plt.title('Корреляция с исходным сигналом')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# График фактора сглаживания
plt.subplot(2, 2, 3)
plt.bar(periods, smoothing_factors, color='lightgreen')
plt.xlabel('Период фильтрации')
plt.ylabel('Фактор сглаживания')
plt.title('Фактор сглаживания')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# График частотной характеристики фильтров
plt.subplot(2, 2, 4)
f_theoretical = np.logspace(-3, 0, 1000)
colors = ['red', 'orange', 'green', 'purple', 'brown']

for i, (period, T) in enumerate(T_values.items()):
    omega = 2 * np.pi * f_theoretical
    magnitude = 1 / np.sqrt(1 + (omega * T)**2)
    plt.semilogx(f_theoretical, 20 * np.log10(magnitude), color=colors[i], linewidth=2, label=period)

plt.xlabel('Частота (1/день)')
plt.ylabel('АЧХ (дБ)')
plt.title('АЧХ фильтров первого порядка')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-40, 5)

plt.tight_layout()
plt.savefig('../images/task3/stock_data_smoothing_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# График 4: Сравнение методов исправления начального значения
plt.figure(figsize=(15, 10))

T_test = 30  # 1 месяц
alpha, beta = create_first_order_filter(T_test)

# Метод 1: Начальное значение равно нулю (стандартный lsim)
y_zero = np.zeros_like(prices)
y_zero[0] = alpha * prices[0]
for i in range(1, len(prices)):
    y_zero[i] = alpha * prices[i] + beta * y_zero[i-1]

# Метод 2: Начальное значение равно первому значению входного сигнала
y_first = apply_filter_with_correction(prices, alpha, beta)

# Метод 3: Начальное значение равно среднему значению
y_mean = np.zeros_like(prices)
y_mean[0] = np.mean(prices)
for i in range(1, len(prices)):
    y_mean[i] = alpha * prices[i] + beta * y_mean[i-1]

# Показываем первые 100 дней для сравнения
mask_initial = dates <= dates[100]

plt.subplot(2, 2, 1)
plt.plot(dates[mask_initial], prices[mask_initial], 'b-', linewidth=2, label='Исходные данные')
plt.plot(dates[mask_initial], y_zero[mask_initial], 'r-', linewidth=2, label='Начальное значение = 0')
plt.xlabel('Дата')
plt.ylabel('Стоимость акции (руб.)')
plt.title('Метод 1: Начальное значение = 0')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
plt.plot(dates[mask_initial], prices[mask_initial], 'b-', linewidth=2, label='Исходные данные')
plt.plot(dates[mask_initial], y_first[mask_initial], 'g-', linewidth=2, label='Начальное значение = x[0]')
plt.xlabel('Дата')
plt.ylabel('Стоимость акции (руб.)')
plt.title('Метод 2: Начальное значение = x[0]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
plt.plot(dates[mask_initial], prices[mask_initial], 'b-', linewidth=2, label='Исходные данные')
plt.plot(dates[mask_initial], y_mean[mask_initial], 'purple', linewidth=2, label='Начальное значение = среднее')
plt.xlabel('Дата')
plt.ylabel('Стоимость акции (руб.)')
plt.title('Метод 3: Начальное значение = среднее')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Сравнение ошибок
plt.subplot(2, 2, 4)
methods = ['Начальное = 0', 'Начальное = x[0]', 'Начальное = среднее']
error_values = [
    np.mean((prices - y_zero)**2),
    np.mean((prices - y_first)**2),
    np.mean((prices - y_mean)**2)
]

plt.bar(methods, error_values, color=['red', 'green', 'purple'])
plt.xlabel('Метод')
plt.ylabel('Среднеквадратичная ошибка')
plt.title('Сравнение методов исправления начального значения')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/task3/stock_data_initial_value_correction.png', dpi=300, bbox_inches='tight')
plt.close()

# Вывод статистики
print("Анализ сглаживания биржевых данных:")
print("=" * 50)

for period, T in T_values.items():
    alpha, beta = create_first_order_filter(T)
    filtered_prices = apply_filter_with_correction(prices, alpha, beta)
    
    mse = np.mean((prices - filtered_prices)**2)
    correlation = np.corrcoef(prices, filtered_prices)[0, 1]
    smoothing_factor = np.std(filtered_prices) / np.std(prices)
    
    print(f"{period}:")
    print(f"  Постоянная времени T = {T} дней")
    print(f"  Среднеквадратичная ошибка: {mse:.2f}")
    print(f"  Корреляция с исходным сигналом: {correlation:.4f}")
    print(f"  Фактор сглаживания: {smoothing_factor:.4f}")
    print()

print("Рекомендации по выбору постоянной времени:")
print("- 1 день: минимальное сглаживание, сохранение краткосрочных колебаний")
print("- 1 неделя: умеренное сглаживание, подходит для краткосрочного анализа")
print("- 1 месяц: сильное сглаживание, выявление среднесрочных трендов")
print("- 3 месяца: очень сильное сглаживание, долгосрочные тренды")
print("- 1 год: максимальное сглаживание, долгосрочные циклы") 