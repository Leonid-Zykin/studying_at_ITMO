import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import symbols, lambdify
import warnings
warnings.filterwarnings('ignore')

# Настройка для отображения русских символов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def plot_3d_phase_portrait():
    """Строит 3D фазовый портрет для системы 7"""
    print("Построение 3D фазового портрета для системы 7...")
    
    x1, x2, x3 = symbols('x1 x2 x3', real=True)
    
    # Система 7: ẋ₁ = -x₁³ + x₂³, ẋ₂ = x₁ + 3x₃ - x₂³, ẋ₃ = x₁x₃ - x₂³ - sin x₁
    f1 = -x1**3 + x2**3
    f2 = x1 + 3*x3 - x2**3
    f3 = x1*x3 - x2**3 - sp.sin(x1)
    
    # Создаем функции для численного интегрирования
    f1_func = lambdify([x1, x2, x3], f1, 'numpy')
    f2_func = lambdify([x1, x2, x3], f2, 'numpy')
    f3_func = lambdify([x1, x2, x3], f3, 'numpy')
    
    def system(t, y):
        x1_val, x2_val, x3_val = y
        return [f1_func(x1_val, x2_val, x3_val), 
               f2_func(x1_val, x2_val, x3_val), 
               f3_func(x1_val, x2_val, x3_val)]
    
    # Создаем 3D график
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Точка равновесия (0, 0, 0)
    ax.scatter(0, 0, 0, c='red', s=100, label='Точка равновесия (0,0,0)')
    
    # Начальные условия для траекторий
    initial_conditions = [
        [0.1, 0.1, 0.1], [-0.1, -0.1, -0.1], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2],
        [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [1, 1, 1], [-1, -1, -1],
        [0.3, 0, 0.3], [0, 0.3, 0.3], [0.3, 0.3, 0]
    ]
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
    
    print(f"Интегрируем {len(initial_conditions)} траекторий...")
    
    for i, ic in enumerate(initial_conditions):
        try:
            # Интегрируем траекторию
            t_span = (0, 5)  # Уменьшаем время для стабильности
            t_eval = np.linspace(0, 5, 500)
            
            sol = solve_ivp(system, t_span, ic, t_eval=t_eval, rtol=1e-8, atol=1e-10)
            
            if sol.success:
                ax.plot(sol.y[0], sol.y[1], sol.y[2], 
                       color=colors[i % len(colors)], alpha=0.7, linewidth=1,
                       label=f'Траектория {i+1}' if i < 5 else "")
                print(f"Траектория {i+1}: успешно интегрирована")
            else:
                print(f"Траектория {i+1}: ошибка интегрирования")
                
        except Exception as e:
            print(f"Траектория {i+1}: ошибка - {e}")
            continue
    
    # Настройка осей
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('x₃')
    ax.set_title('3D Фазовый портрет системы 7\nẋ₁ = -x₁³ + x₂³, ẋ₂ = x₁ + 3x₃ - x₂³, ẋ₃ = x₁x₃ - x₂³ - sin x₁')
    
    # Устанавливаем пределы осей
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    
    # Добавляем легенду
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Сохраняем изображение
    filename = '/home/leonidas/projects/studying_at_ITMO/nonlinear_systems/lab1/python/system7_3d_phase_portrait.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"3D фазовый портрет сохранен: {filename}")
    
    plt.show()
    
    return fig

def plot_2d_projections():
    """Строит 2D проекции 3D фазового портрета"""
    print("Построение 2D проекций...")
    
    x1, x2, x3 = symbols('x1 x2 x3', real=True)
    
    # Система 7
    f1 = -x1**3 + x2**3
    f2 = x1 + 3*x3 - x2**3
    f3 = x1*x3 - x2**3 - sp.sin(x1)
    
    # Создаем функции
    f1_func = lambdify([x1, x2, x3], f1, 'numpy')
    f2_func = lambdify([x1, x2, x3], f2, 'numpy')
    f3_func = lambdify([x1, x2, x3], f3, 'numpy')
    
    def system(t, y):
        x1_val, x2_val, x3_val = y
        return [f1_func(x1_val, x2_val, x3_val), 
               f2_func(x1_val, x2_val, x3_val), 
               f3_func(x1_val, x2_val, x3_val)]
    
    # Создаем 2x2 сетку для проекций
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Начальные условия
    initial_conditions = [
        [0.1, 0.1, 0.1], [-0.1, -0.1, -0.1], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2],
        [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]
    ]
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, ic in enumerate(initial_conditions):
        try:
            t_span = (0, 5)
            t_eval = np.linspace(0, 5, 500)
            sol = solve_ivp(system, t_span, ic, t_eval=t_eval, rtol=1e-8, atol=1e-10)
            
            if sol.success:
                # Проекция x1-x2
                axes[0, 0].plot(sol.y[0], sol.y[1], color=colors[i % len(colors)], alpha=0.7, linewidth=1)
                # Проекция x1-x3
                axes[0, 1].plot(sol.y[0], sol.y[2], color=colors[i % len(colors)], alpha=0.7, linewidth=1)
                # Проекция x2-x3
                axes[1, 0].plot(sol.y[1], sol.y[2], color=colors[i % len(colors)], alpha=0.7, linewidth=1)
                # Проекция x1-x2-x3 (3D вид)
                axes[1, 1].plot(sol.y[0], sol.y[1], color=colors[i % len(colors)], alpha=0.7, linewidth=1)
                
        except:
            continue
    
    # Настройка подписей
    axes[0, 0].set_xlabel('x₁')
    axes[0, 0].set_ylabel('x₂')
    axes[0, 0].set_title('Проекция x₁-x₂')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('x₁')
    axes[0, 1].set_ylabel('x₃')
    axes[0, 1].set_title('Проекция x₁-x₃')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('x₂')
    axes[1, 0].set_ylabel('x₃')
    axes[1, 0].set_title('Проекция x₂-x₃')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('x₁')
    axes[1, 1].set_ylabel('x₂')
    axes[1, 1].set_title('Проекция x₁-x₂ (общий вид)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('2D Проекции 3D фазового портрета системы 7', fontsize=16)
    plt.tight_layout()
    
    # Сохраняем изображение
    filename = '/home/leonidas/projects/studying_at_ITMO/nonlinear_systems/lab1/python/system7_2d_projections.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"2D проекции сохранены: {filename}")
    
    plt.show()
    
    return fig

def main():
    """Основная функция"""
    print("=" * 80)
    print("ПОСТРОЕНИЕ ФАЗОВЫХ ПОРТРЕТОВ ДЛЯ СИСТЕМЫ 7 (3D)")
    print("=" * 80)
    
    # Строим 3D фазовый портрет
    fig_3d = plot_3d_phase_portrait()
    
    # Строим 2D проекции
    fig_2d = plot_2d_projections()
    
    print("\n" + "=" * 80)
    print("ПОСТРОЕНИЕ ЗАВЕРШЕНО!")
    print("Созданы файлы:")
    print("- system7_3d_phase_portrait.png")
    print("- system7_2d_projections.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
