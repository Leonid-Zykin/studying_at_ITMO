import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

def sinc_function(t, a=1, b=1):
    # нормализованный sinc (sin(pi x)/(pi x))
    return a * np.sinc(b * t)

def fourier_transform_sinc(omega, a=1, b=1):
    # спектр — прямоугольная функция шириной 2*pi*b
    return np.where(np.abs(omega) <= np.pi * b, a / np.sqrt(2 * np.pi), 0)

a = 1
b_values = [0.5, 1, 2]

t = np.linspace(-10, 10, 2000)
omega = np.linspace(-20, 20, 2000)

for b in b_values:
    f_t = sinc_function(t, a, b)
    f_omega = fourier_transform_sinc(omega, a, b)

    # f(t)
    plt.figure(figsize=(12, 4))
    plt.plot(t, f_t)
    plt.title(f"sinc-функция f(t), a={a}, b={b}")
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"sinc_function_b{b}.png")
    plt.show()

    # f^(ω)
    plt.figure(figsize=(12, 4))
    plt.plot(omega, f_omega)
    plt.title(f"Фурье-образ sinc-функции $\\hat{{f}}(\\omega)$, a={a}, b={b}")
    plt.xlabel("ω")
    plt.ylabel("f̂(ω)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"sinc_fourier_b{b}.png")
    plt.show()

    # Парсеваль
    lhs = simps(np.abs(f_t)**2, t)
    rhs = simps(np.abs(f_omega)**2, omega)
    print(f"a = {a}, b = {b}")
    print(f"Интеграл |f(t)|^2 dt = {lhs:.4f}")
    print(f"Интеграл |f̂(ω)|^2 dω = {rhs:.4f}")
    print(f"Разность = {abs(lhs - rhs):.4e}")
    print("-" * 40) 