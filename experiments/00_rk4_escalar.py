"""Validación del integrador RK4 con la ecuación ẋ = x (solución exacta: e^t)."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def rk4_step(f, t, x, dt):
    """Un paso de RK4 para ẋ = f(t, x).

    f  : función que recibe (t, x) y devuelve la derivada
    t  : instante actual
    x  : estado actual
    dt : tamaño del paso
    """
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt * k1 / 2)
    k3 = f(t + dt / 2, x + dt * k2 / 2)
    k4 = f(t + dt, x + dt * k3)

    return x + (dt / 6) * (k1 + 2 * (k2 + k3) + k4)


def f(t, x):
    """Dinámica de prueba: ẋ = x."""
    return x


def simular(f, x0, T, dt):
    """Integra de 0 a T con RK4. Devuelve (tiempos, estados)."""
    N = int(round(T / dt))
    ts = np.linspace(0.0, T, N + 1)
    xs = np.zeros(N + 1)
    xs[0] = x0

    for k in range(N):
        xs[k + 1] = rk4_step(f, ts[k], xs[k], dt)

    return ts, xs


# --- Parámetros de la simulación ---
x0 = 1.0
T = 1.0
dt = 0.01

ts, xs = simular(f, x0, T, dt)

# --- Error en el instante final ---
exacto = np.exp(T)
print(f"Exacto    : {exacto:.10f}")
print(f"RK4       : {xs[-1]:.10f}   error = {abs(xs[-1] - exacto):.3e}")

# --- Comparación con solve_ivp ---
sol = solve_ivp(f, [0, T], [x0], t_eval=ts, rtol=1e-10, atol=1e-12)
x_scipy = sol.y[0]

print(f"solve_ivp : {x_scipy[-1]:.10f}   error = {abs(x_scipy[-1] - exacto):.3e}")
print(f"Diferencia RK4 vs solve_ivp: {abs(xs[-1] - x_scipy[-1]):.3e}")

# --- Gráfica ---
t_fino = np.linspace(0, T, 300)

fig, ax = plt.subplots()
ax.plot(t_fino, np.exp(t_fino), "k-", label="exacta ($e^t$)")
ax.plot(ts, xs, "o", label="RK4", markersize=4)
ax.plot(ts, x_scipy, "x", label="solve_ivp", markersize=4)
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_title(f"Solución (dt = {dt})")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()