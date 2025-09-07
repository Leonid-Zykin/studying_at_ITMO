import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import symbols, solve, diff, Matrix, lambdify
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

# Настройка для отображения русских символов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def plot_2d_phase_portrait(f1_func, f2_func, eq_points, title, filename, x1_range=(-3, 3), x2_range=(-3, 3)):
    """Строит 2D фазовый портрет системы"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Создаем сетку для построения векторного поля
    x1 = np.linspace(x1_range[0], x1_range[1], 20)
    x2 = np.linspace(x2_range[0], x2_range[1], 20)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Вычисляем производные в каждой точке сетки
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
    
    # Отмечаем точки равновесия
    for i, ep in enumerate(eq_points):
        ax.plot(ep[0], ep[1], 'ro', markersize=8, label=f'Точка равновесия {i+1}' if i == 0 else "")
    
    # Добавляем несколько траекторий
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 1000)
    
    # Начальные условия для траекторий
    initial_conditions = [
        [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5],
        [1.5, 1.5], [-1.5, 1.5], [1.5, -1.5], [-1.5, -1.5]
    ]
    
    for ic in initial_conditions:
        try:
            sol = solve_ivp(
                lambda t, y: [f1_func(y[0], y[1]), f2_func(y[0], y[1])],
                t_span, ic, t_eval=t_eval, rtol=1e-8, atol=1e-10
            )
            if sol.success:
                ax.plot(sol.y[0], sol.y[1], 'b-', alpha=0.7, linewidth=1)
        except:
            continue
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(x1_range)
    ax.set_ylim(x2_range)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Фазовый портрет сохранен: {filename}")

def plot_3d_phase_portrait(f1_func, f2_func, f3_func, eq_points, title, filename, x1_range=(-2, 2), x2_range=(-2, 2), x3_range=(-2, 2)):
    """Строит 3D фазовый портрет системы"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Отмечаем точки равновесия
    for i, ep in enumerate(eq_points):
        ax.scatter(ep[0], ep[1], ep[2], c='red', s=100, label=f'Точка равновесия {i+1}' if i == 0 else "")
    
    # Добавляем несколько траекторий
    t_span = (0, 5)
    t_eval = np.linspace(0, 5, 500)
    
    # Начальные условия для траекторий
    initial_conditions = [
        [0.1, 0.1, 0.1], [-0.1, -0.1, -0.1], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2],
        [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [1, 1, 1], [-1, -1, -1]
    ]
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, ic in enumerate(initial_conditions):
        try:
            sol = solve_ivp(
                lambda t, y: [f1_func(y[0], y[1], y[2]), 
                             f2_func(y[0], y[1], y[2]), 
                             f3_func(y[0], y[1], y[2])],
                t_span, ic, t_eval=t_eval, rtol=1e-8, atol=1e-10
            )
            if sol.success:
                ax.plot(sol.y[0], sol.y[1], sol.y[2], 
                       color=colors[i % len(colors)], alpha=0.7, linewidth=1)
        except:
            continue
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('x₃')
    ax.set_title(title)
    ax.legend()
    
    # Устанавливаем пределы осей
    ax.set_xlim(x1_range)
    ax.set_ylim(x2_range)
    ax.set_zlim(x3_range)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"3D фазовый портрет сохранен: {filename}")

def analyze_system_with_plots(system_name, f1_expr, f2_expr, f3_expr=None):
    """Анализ системы с построением фазовых портретов"""
    print(f"\n=== АНАЛИЗ {system_name} ===")
    
    x1, x2, x3 = symbols('x1 x2 x3', real=True)
    eq_points = []
    
    if f3_expr is None:
        # 2D система
        print("Поиск точек равновесия 2D системы...")
        
        # 1. Аналитическое решение
        try:
            solutions = solve([f1_expr, f2_expr], [x1, x2], dict=True)
            for sol in solutions:
                if x1 in sol and x2 in sol:
                    x1_val = float(sol[x1].evalf())
                    x2_val = float(sol[x2].evalf())
                    eq_points.append([x1_val, x2_val])
            print(f"Аналитически найдено {len(eq_points)} точек")
        except:
            print("Аналитическое решение невозможно")
        
        # 2. Если точек мало, дополняем численным поиском
        if len(eq_points) < 3:
            print("Дополняем численным поиском...")
            f1_func = lambdify([x1, x2], f1_expr, 'numpy')
            f2_func = lambdify([x1, x2], f2_expr, 'numpy')
            
            def system(vars):
                x1_val, x2_val = vars
                return [f1_func(x1_val, x2_val), f2_func(x1_val, x2_val)]
            
            # Поиск только в ключевых точках
            key_points = [
                [0, 0], [1, 1], [-1, -1], [1, -1], [-1, 1],
                [0.5, 0.5], [-0.5, -0.5], [2, 2], [-2, -2]
            ]
            
            for point in key_points:
                try:
                    sol = fsolve(system, point, xtol=1e-8)
                    if (abs(system(sol)[0]) < 1e-6 and abs(system(sol)[1]) < 1e-6):
                        is_duplicate = False
                        for ep in eq_points:
                            if np.linalg.norm(np.array(sol) - np.array(ep)) < 1e-4:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            eq_points.append(sol)
                except:
                    continue
        
        print(f"Всего найдено точек равновесия: {len(eq_points)}")
        
        # Анализируем каждую точку
        for i, ep in enumerate(eq_points):
            print(f"Точка {i+1}: ({ep[0]:.6f}, {ep[1]:.6f})")
            
            # Линеаризуем систему
            J11 = diff(f1_expr, x1).subs({x1: ep[0], x2: ep[1]})
            J12 = diff(f1_expr, x2).subs({x1: ep[0], x2: ep[1]})
            J21 = diff(f2_expr, x1).subs({x1: ep[0], x2: ep[1]})
            J22 = diff(f2_expr, x2).subs({x1: ep[0], x2: ep[1]})
            
            J = Matrix([[J11, J12], [J21, J22]])
            print(f"Матрица Якоби:\n{J}")
            
            # Анализируем собственные значения
            eigenvals = J.eigenvals()
            eigenvals_list = list(eigenvals.keys())
            eigenvals_numeric = [complex(ev.evalf()) for ev in eigenvals_list]
            real_parts = [ev.real for ev in eigenvals_numeric]
            imag_parts = [ev.imag for ev in eigenvals_numeric]
            
            print(f"Собственные значения: {eigenvals_numeric}")
            
            # Определяем тип
            if all(r < 0 for r in real_parts):
                if all(abs(i) < 1e-10 for i in imag_parts):
                    eq_type = "Устойчивый узел"
                else:
                    eq_type = "Устойчивый фокус"
            elif all(r > 0 for r in real_parts):
                if all(abs(i) < 1e-10 for i in imag_parts):
                    eq_type = "Неустойчивый узел"
                else:
                    eq_type = "Неустойчивый фокус"
            elif any(r < 0 for r in real_parts) and any(r > 0 for r in real_parts):
                eq_type = "Седло"
            else:
                eq_type = "Центр или вырожденный случай"
            
            print(f"Тип точки равновесия: {eq_type}")
            print("-" * 50)
        
        # Строим фазовый портрет
        f1_func = lambdify([x1, x2], f1_expr, 'numpy')
        f2_func = lambdify([x1, x2], f2_expr, 'numpy')
        
        system_num = system_name.split()[-1].replace('(', '').replace(')', '')
        filename = f'/home/leonidas/projects/studying_at_ITMO/nonlinear_systems/lab1/python/system{system_num}_phase_portrait.png'
        title = f"Система {system_num}: {system_name}"
        
        plot_2d_phase_portrait(f1_func, f2_func, eq_points, title, filename)
        
        return eq_points
    
    else:
        # 3D система
        print("Поиск точек равновесия 3D системы...")
        
        # Сразу переходим к численным методам для 3D систем
        print("Используем численные методы...")
        
        # Создаем функции для численного поиска
        f1_func = lambdify([x1, x2, x3], f1_expr, 'numpy')
        f2_func = lambdify([x1, x2, x3], f2_expr, 'numpy')
        f3_func = lambdify([x1, x2, x3], f3_expr, 'numpy')
        
        def system(vars):
            x1_val, x2_val, x3_val = vars
            return [f1_func(x1_val, x2_val, x3_val), 
                   f2_func(x1_val, x2_val, x3_val), 
                   f3_func(x1_val, x2_val, x3_val)]
        
        # Очень ограниченный набор ключевых точек
        key_points = [
            [0, 0, 0], [1, 1, 1], [-1, -1, -1], [0, 1, 0], [1, 0, 0], [0, 0, 1]
        ]
        
        print(f"Проверяем {len(key_points)} ключевых точек...")
        
        for i, point in enumerate(key_points):
            try:
                sol = fsolve(system, point, xtol=1e-8)
                
                if (abs(system(sol)[0]) < 1e-6 and
                    abs(system(sol)[1]) < 1e-6 and
                    abs(system(sol)[2]) < 1e-6):
                    
                    is_duplicate = False
                    for ep in eq_points:
                        if np.linalg.norm(np.array(sol) - np.array(ep)) < 1e-4:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        eq_points.append(sol)
                        print(f"Найдена точка {len(eq_points)}: ({sol[0]:.6f}, {sol[1]:.6f}, {sol[2]:.6f})")
            except:
                continue
        
        print(f"Всего найдено точек равновесия: {len(eq_points)}")
        
        # Анализируем каждую точку
        for i, ep in enumerate(eq_points):
            print(f"Точка {i+1}: ({ep[0]:.6f}, {ep[1]:.6f}, {ep[2]:.6f})")
            
            # Линеаризуем систему
            J11 = diff(f1_expr, x1).subs({x1: ep[0], x2: ep[1], x3: ep[2]})
            J12 = diff(f1_expr, x2).subs({x1: ep[0], x2: ep[1], x3: ep[2]})
            J13 = diff(f1_expr, x3).subs({x1: ep[0], x2: ep[1], x3: ep[2]})
            J21 = diff(f2_expr, x1).subs({x1: ep[0], x2: ep[1], x3: ep[2]})
            J22 = diff(f2_expr, x2).subs({x1: ep[0], x2: ep[1], x3: ep[2]})
            J23 = diff(f2_expr, x3).subs({x1: ep[0], x2: ep[1], x3: ep[2]})
            J31 = diff(f3_expr, x1).subs({x1: ep[0], x2: ep[1], x3: ep[2]})
            J32 = diff(f3_expr, x2).subs({x1: ep[0], x2: ep[1], x3: ep[2]})
            J33 = diff(f3_expr, x3).subs({x1: ep[0], x2: ep[1], x3: ep[2]})
            
            J = Matrix([[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]])
            print(f"Матрица Якоби:\n{J}")
            
            # Анализируем собственные значения
            eigenvals = J.eigenvals()
            eigenvals_list = list(eigenvals.keys())
            eigenvals_numeric = [complex(ev.evalf()) for ev in eigenvals_list]
            real_parts = [ev.real for ev in eigenvals_numeric]
            
            print(f"Собственные значения: {eigenvals_numeric}")
            
            # Определяем тип
            if all(r < 0 for r in real_parts):
                eq_type = "Устойчивый узел/фокус"
            elif all(r > 0 for r in real_parts):
                eq_type = "Неустойчивый узел/фокус"
            elif any(r < 0 for r in real_parts) and any(r > 0 for r in real_parts):
                eq_type = "Седло"
            else:
                eq_type = "Центр или вырожденный случай"
            
            print(f"Тип точки равновесия: {eq_type}")
            print("-" * 50)
        
        # Строим 3D фазовый портрет
        system_num = system_name.split()[-1].replace('(', '').replace(')', '')
        filename = f'/home/leonidas/projects/studying_at_ITMO/nonlinear_systems/lab1/python/system{system_num}_3d_phase_portrait.png'
        title = f"Система {system_num} (3D): {system_name}"
        
        plot_3d_phase_portrait(f1_func, f2_func, f3_func, eq_points, title, filename)
        
        return eq_points

def main():
    """Основная функция с построением всех фазовых портретов"""
    print("=" * 80)
    print("ПОЛНЫЙ АНАЛИЗ С ФАЗОВЫМИ ПОРТРЕТАМИ")
    print("=" * 80)
    
    x1, x2, x3 = symbols('x1 x2 x3', real=True)
    
    # Система 1: ẋ₁ = -x₁ + 2x₁³ + x₂, ẋ₂ = -x₁ - x₂
    f1_1 = -x1 + 2*x1**3 + x2
    f2_1 = -x1 - x2
    eq1 = analyze_system_with_plots("СИСТЕМА 1", f1_1, f2_1)
    
    # Система 2: ẋ₁ = x₁ + x₁x₂, ẋ₂ = -x₂ + x₂² + x₁x₂ - x₁³
    f1_2 = x1 + x1*x2
    f2_2 = -x2 + x2**2 + x1*x2 - x1**3
    eq2 = analyze_system_with_plots("СИСТЕМА 2", f1_2, f2_2)
    
    # Система 3: ẋ₁ = x₂, ẋ₂ = -x₁ + x₂(1 - x₁² + 0.1x₁⁴)
    f1_3 = x2
    f2_3 = -x1 + x2*(1 - x1**2 + 0.1*x1**4)
    eq3 = analyze_system_with_plots("СИСТЕМА 3", f1_3, f2_3)
    
    # Система 4: ẋ₁ = (x₁ - x₂)(1 - x₁² - x₂²), ẋ₂ = (x₁ + x₂)(1 - x₁² - x₂²)
    f1_4 = (x1 - x2)*(1 - x1**2 - x2**2)
    f2_4 = (x1 + x2)*(1 - x1**2 - x2**2)
    eq4 = analyze_system_with_plots("СИСТЕМА 4", f1_4, f2_4)
    
    # Система 5: ẋ₁ = -x₁³ + x₂, ẋ₂ = x₁ - x₂³
    f1_5 = -x1**3 + x2
    f2_5 = x1 - x2**3
    eq5 = analyze_system_with_plots("СИСТЕМА 5", f1_5, f2_5)
    
    # Система 6: ẋ₁ = -x₁³ + x₂³, ẋ₂ = x₂³x₁ - x₂³
    f1_6 = -x1**3 + x2**3
    f2_6 = x2**3*x1 - x2**3
    eq6 = analyze_system_with_plots("СИСТЕМА 6", f1_6, f2_6)
    
    # Система 7: 3D система
    f1_7 = -x1**3 + x2**3
    f2_7 = x1 + 3*x3 - x2**3
    f3_7 = x1*x3 - x2**3 - sp.sin(x1)
    eq7 = analyze_system_with_plots("СИСТЕМА 7 (3D)", f1_7, f2_7, f3_7)
    
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:")
    print("-" * 80)
    print(f"{'Система':<15} {'Точек равновесия':<18}")
    print("-" * 80)
    print(f"{'Система 1':<15} {len(eq1):<18}")
    print(f"{'Система 2':<15} {len(eq2):<18}")
    print(f"{'Система 3':<15} {len(eq3):<18}")
    print(f"{'Система 4':<15} {len(eq4):<18}")
    print(f"{'Система 5':<15} {len(eq5):<18}")
    print(f"{'Система 6':<15} {len(eq6):<18}")
    print(f"{'Система 7':<15} {len(eq7):<18}")
    print("=" * 80)
    
    print("\nСОЗДАННЫЕ ИЗОБРАЖЕНИЯ:")
    print("- system1_phase_portrait.png")
    print("- system2_phase_portrait.png")
    print("- system3_phase_portrait.png")
    print("- system4_phase_portrait.png")
    print("- system5_phase_portrait.png")
    print("- system6_phase_portrait.png")
    print("- system7_3d_phase_portrait.png")

if __name__ == "__main__":
    main()
