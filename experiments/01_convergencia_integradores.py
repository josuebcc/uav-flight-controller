"""Compara Euler, Heun y RK4 con dx/dt = x, cuya solución es x(t) = exp(t)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uav_sim.integrators.one_step import euler_step, heun_step, rk4_step


def exponencial(x, u):
    """Devuelve la derivada dx/dt = x. La entrada u no se usa en este ejemplo."""
    return x


def integrar(solver, f, x0, u, tiempo_total, h):
    """Repite un método de un paso y devuelve los tiempos y los estados calculados."""
    numero_de_pasos = round(tiempo_total / h)
    if not np.isclose(numero_de_pasos * h, tiempo_total):
        raise ValueError("El tiempo total debe ser un múltiplo del paso h.")
    x = np.array(x0, dtype=float)
    xk = [x]
    for _ in range(numero_de_pasos):
        x = solver(f, x, u, h)
        xk.append(x)

    tiempos = np.arange(numero_de_pasos + 1) * h
    return tiempos, np.array(xk)


# 1. Parámetros: empezamos en x = 1 y simulamos hasta t = 1.
TIEMPO_TOTAL = 1.0
ESTADO_INICIAL = [1.0]
# Cada paso es diez veces menor que el anterior.
PASOS = [1, 0.1, 0.01, 0.001, 0.0001]
METODOS = {"Euler": euler_step, "Heun": heun_step, "RK4": rk4_step}

# 2. Medir el error final de cada método al reducir el paso h.
valor_exacto = np.exp(TIEMPO_TOTAL)
errores = {}
for nombre, solver in METODOS.items():
    errores[nombre] = []
    for h in PASOS:
        tiempos, xk = integrar(solver, exponencial, ESTADO_INICIAL, 0.0, TIEMPO_TOTAL, h)
        valor_final = xk[-1, 0]  # Último instante, único componente del estado.
        errores[nombre].append(abs(valor_final - valor_exacto))

# 3. Representar el error frente a h y calcular el orden de cada método.
# En escala log-log, la pendiente se aproxima al orden: Euler 1, Heun 2 y RK4 4.
fig_convergencia, ax = plt.subplots()
for nombre, errores_metodo in errores.items():
    orden, _ = np.polyfit(np.log(PASOS), np.log(errores_metodo), 1)
    print(f"{nombre}: orden medido = {orden:.2f}")
    ax.loglog(PASOS, errores_metodo, "o-", label=f"{nombre} (orden {orden:.2f})")

ax.set_title("Convergencia de los integradores")
ax.set_xlabel("Paso h")
ax.set_ylabel(f"Error absoluto en t = {TIEMPO_TOTAL:g}")
ax.grid(True)
ax.legend()

# 4. Guardar la gráfica junto al experimento y mostrarla.
carpeta_salida = Path(__file__).parent / "salidas"
carpeta_salida.mkdir(exist_ok=True)
fig_convergencia.savefig(carpeta_salida / "01_convergencia.png", dpi=300)
print(f"Gráfica guardada en {carpeta_salida}")
plt.show()
