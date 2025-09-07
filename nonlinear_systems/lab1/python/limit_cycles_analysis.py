import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import symbols, diff, lambdify
import warnings
warnings.filterwarnings('ignore')

# Настройка для отображения русских символов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_limit_cycles_polar(system_name, f1_expr, f2_expr):
    """Анализ предельных циклов с использованием полярных координат"""
    print(f"\n=== АНАЛИЗ ПРЕДЕЛЬНЫХ ЦИКЛОВ {system_name} ===")
    
    x1, x2 = symbols('x1 x2', real=True)
    
    # Переход к полярным координатам
    r, theta = symbols('r theta', real=True, positive=True)
    
    # Преобразования: x1 = r*cos(theta), x2 = r*sin(theta)
    x1_polar = r * sp.cos(theta)
    x2_polar = r * sp.sin(theta)
    
    # Подставляем в исходную систему
    f1_polar = f1_expr.subs({x1: x1_polar, x2: x2_polar})
    f2_polar = f2_expr.subs({x1: x1_polar, x2: x2_polar})
    
    # Вычисляем производные по времени
    # dr/dt = (x1*dx1/dt + x2*dx2/dt)/r
    # dtheta/dt = (x1*dx2/dt - x2*dx1/dt)/r^2
    
    dr_dt = (x1_polar * f1_polar + x2_polar * f2_polar) / r
    dtheta_dt = (x1_polar * f2_polar - x2_polar * f1_polar) / (r**2)
    
    print("Система в полярных координатах:")
    print(f"dr/dt = {dr_dt.simplify()}")
    print(f"dtheta/dt = {dtheta_dt.simplify()}")
    
    # Ищем предельные циклы: dr/dt = 0 при r > 0
    print("\nПоиск предельных циклов (dr/dt = 0):")
    
    try:
        # Решаем dr/dt = 0 относительно r
        limit_cycles = sp.solve(dr_dt, r)
        print(f"Найдены решения: {limit_cycles}")
        
        # Фильтруем положительные решения
        positive_cycles = []
        for cycle in limit_cycles:
            if cycle.is_positive:
                positive_cycles.append(cycle)
        
        print(f"Положительные предельные циклы: {positive_cycles}")
        
        # Анализируем устойчивость каждого цикла
        for i, cycle_r in enumerate(positive_cycles):
            print(f"\nАнализ цикла {i+1}: r = {cycle_r}")
            
            # Вычисляем производную dr/dt по r в точке цикла
            dr_dr = diff(dr_dt, r).subs(r, cycle_r)
            print(f"d(dr/dt)/dr в точке цикла: {dr_dr}")
            
            if dr_dr < 0:
                print("Устойчивый предельный цикл")
            elif dr_dr > 0:
                print("Неустойчивый предельный цикл")
            else:
                print("Нейтральный предельный цикл (требует дополнительного анализа)")
                
    except Exception as e:
        print(f"Ошибка при анализе предельных циклов: {e}")
    
    # Строим фазовый портрет для визуализации
    f1_func = lambdify([x1, x2], f1_expr, 'numpy')
    f2_func = lambdify([x1, x2], f2_expr, 'numpy')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Создаем сетку для векторного поля
    x1_range = np.linspace(-3, 3, 20)
    x2_range = np.linspace(-3, 3, 20)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    U = np.zeros_like(X1)
    V = np.zeros_like(X2)
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            try:
                U[i, j] = f1_func(X1[i, j], X2[i, j])
                V[i, j] = f2_func(X1[i, j], X2[i, j])
            except:
                U[i, j] = 0
                V[i, j] = 0
    
    # Строим векторное поле
    ax.quiver(X1, X2, U, V, alpha=0.6, scale=50)
    
    # Добавляем траектории
    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 2000)
    
    # Начальные условия на окружностях разного радиуса
    for r_init in [0.5, 1.0, 1.5, 2.0]:
        for theta_init in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            x1_init = r_init * np.cos(theta_init)
            x2_init = r_init * np.sin(theta_init)
            
            try:
                sol = solve_ivp(
                    lambda t, y: [f1_func(y[0], y[1]), f2_func(y[0], y[1])],
                    t_span, [x1_init, x2_init], t_eval=t_eval, rtol=1e-8, atol=1e-10
                )
                if sol.success:
                    ax.plot(sol.y[0], sol.y[1], 'b-', alpha=0.7, linewidth=1)
            except:
                continue
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(f'Фазовый портрет {system_name} с анализом предельных циклов')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    # Добавляем окружности для визуализации предельных циклов
    for r_val in [0.5, 1.0, 1.5, 2.0]:
        circle = plt.Circle((0, 0), r_val, fill=False, color='red', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
    
    plt.tight_layout()
    filename = f'/home/leonidas/projects/studying_at_ITMO/nonlinear_systems/lab1/python/limit_cycles_{system_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Фазовый портрет с анализом предельных циклов сохранен: {filename}")

def main():
    """Анализ предельных циклов для всех систем"""
    print("=" * 80)
    print("АНАЛИЗ ПРЕДЕЛЬНЫХ ЦИКЛОВ С ИСПОЛЬЗОВАНИЕМ ПОЛЯРНЫХ КООРДИНАТ")
    print("=" * 80)
    
    x1, x2 = symbols('x1 x2', real=True)
    
    # Система 3: ẋ₁ = x₂, ẋ₂ = -x₁ + x₂(1 - x₁² + 0.1x₁⁴)
    f1_3 = x2
    f2_3 = -x1 + x2*(1 - x1**2 + 0.1*x1**4)
    analyze_limit_cycles_polar("СИСТЕМА 3", f1_3, f2_3)
    
    # Система 4: ẋ₁ = (x₁ - x₂)(1 - x₁² - x₂²), ẋ₂ = (x₁ + x₂)(1 - x₁² - x₂²)
    f1_4 = (x1 - x2)*(1 - x1**2 - x2**2)
    f2_4 = (x1 + x2)*(1 - x1**2 - x2**2)
    analyze_limit_cycles_polar("СИСТЕМА 4", f1_4, f2_4)
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ ПРЕДЕЛЬНЫХ ЦИКЛОВ ЗАВЕРШЕН!")
    print("=" * 80)

if __name__ == "__main__":
    main()
