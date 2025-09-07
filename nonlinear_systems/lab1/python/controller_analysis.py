import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import sympy as sp
from sympy import symbols, diff, Matrix, lambdify
import warnings
warnings.filterwarnings('ignore')

# Настройка для отображения русских символов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_controlled_system(system_name, f1_expr, f2_expr, u1_expr, u2_expr):
    """Анализ управляемой системы"""
    print(f"\n=== АНАЛИЗ УПРАВЛЯЕМОЙ СИСТЕМЫ {system_name} ===")
    
    x1, x2, u1, u2 = symbols('x1 x2 u1 u2', real=True)
    
    # Находим точки равновесия системы без управления
    f1_uncontrolled = f1_expr.subs({u1: 0, u2: 0})
    f2_uncontrolled = f2_expr.subs({u1: 0, u2: 0})
    
    print("Поиск точек равновесия системы без управления...")
    eq_points = []
    
    try:
        solutions = sp.solve([f1_uncontrolled, f2_uncontrolled], [x1, x2], dict=True)
        for sol in solutions:
            if x1 in sol and x2 in sol:
                x1_val = float(sol[x1].evalf())
                x2_val = float(sol[x2].evalf())
                eq_points.append([x1_val, x2_val])
        print(f"Найдено {len(eq_points)} точек равновесия")
    except:
        print("Аналитическое решение невозможно")
    
    # Анализируем каждую точку и синтезируем регулятор
    for i, ep in enumerate(eq_points):
        print(f"\nТочка равновесия {i+1}: ({ep[0]:.6f}, {ep[1]:.6f})")
        
        # Линеаризуем систему в этой точке
        A11 = diff(f1_expr, x1).subs({x1: ep[0], x2: ep[1], u1: 0, u2: 0})
        A12 = diff(f1_expr, x2).subs({x1: ep[0], x2: ep[1], u1: 0, u2: 0})
        A21 = diff(f2_expr, x1).subs({x1: ep[0], x2: ep[1], u1: 0, u2: 0})
        A22 = diff(f2_expr, x2).subs({x1: ep[0], x2: ep[1], u1: 0, u2: 0})
        
        B11 = diff(f1_expr, u1).subs({x1: ep[0], x2: ep[1], u1: 0, u2: 0})
        B12 = diff(f1_expr, u2).subs({x1: ep[0], x2: ep[1], u1: 0, u2: 0})
        B21 = diff(f2_expr, u1).subs({x1: ep[0], x2: ep[1], u1: 0, u2: 0})
        B22 = diff(f2_expr, u2).subs({x1: ep[0], x2: ep[1], u1: 0, u2: 0})
        
        A = np.array([[float(A11), float(A12)], [float(A21), float(A22)]])
        B = np.array([[float(B11), float(B12)], [float(B21), float(B22)]])
        
        print(f"Матрица A:\n{A}")
        print(f"Матрица B:\n{B}")
        
        # Синтезируем LQR регулятор
        try:
            Q = np.eye(2)  # Матрица весов состояний
            R = np.eye(2)  # Матрица весов управления
            
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            
            print(f"Матрица обратной связи K:\n{K}")
            
            # Проверяем устойчивость замкнутой системы
            A_cl = A - B @ K
            eigenvals = np.linalg.eigvals(A_cl)
            print(f"Собственные значения замкнутой системы: {eigenvals}")
            
            if all(np.real(eigenvals) < 0):
                print("Замкнутая система устойчива")
            else:
                print("Замкнутая система неустойчива")
                
        except Exception as e:
            print(f"Ошибка при синтезе регулятора: {e}")
            # Простой регулятор
            K = -np.eye(2)
            print(f"Используем простой регулятор K:\n{K}")
        
        print("-" * 50)
    
    return eq_points

def main():
    """Основная функция для анализа управляемых систем"""
    print("=" * 80)
    print("АНАЛИЗ УПРАВЛЯЕМЫХ СИСТЕМ И СИНТЕЗ РЕГУЛЯТОРОВ")
    print("=" * 80)
    
    x1, x2, u1, u2 = symbols('x1 x2 u1 u2', real=True)
    
    # Управляемая система 1: ẋ₁ = -x₁ + 2x₁³ + x₂ + sin u₁, ẋ₂ = -x₁ - x₂ + 3 sin u₂
    f1_1 = -x1 + 2*x1**3 + x2 + sp.sin(u1)
    f2_1 = -x1 - x2 + 3*sp.sin(u2)
    eq1 = analyze_controlled_system("1", f1_1, f2_1, sp.sin(u1), 3*sp.sin(u2))
    
    # Управляемая система 2: ẋ₁ = x₂ + x₁x₂ + u₃, ẋ₂ = -x₂ + x₂² - x₁³ + sin u
    # Заменяем u₃ на u₁ и u на u₂ для совместимости
    f1_2 = x2 + x1*x2 + u1
    f2_2 = -x2 + x2**2 - x1**3 + sp.sin(u2)
    eq2 = analyze_controlled_system("2", f1_2, f2_2, u1, sp.sin(u2))
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 80)

if __name__ == "__main__":
    main()
