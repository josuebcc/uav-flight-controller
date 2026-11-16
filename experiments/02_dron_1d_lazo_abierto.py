"""Experimento 02: dron 1D en lazo abierto, simulado con RK4."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uav_sim.dynamics.drone_1d import G, M, drone_1d, empuje_hover
from uav_sim.integrators.one_step import rk4_step
from uav_sim.simulacion import simular

# --- Parámetros ---
H = 0.1  # paso [s]
X0 = np.array([0.0, 0.0])  # empieza a 0 m y quieto
TRAMOS = [  # (duración [s], empuje [N])
    (2.0, 1.3 * empuje_hover()),  # sube acelerando
    (2.0, 0.8 * empuje_hover()),  # frena
    (2.0, 1.0 * empuje_hover()),  # mantiene la velocidad de subida: empuje = peso
]

# --- Perfil de empuje: un valor por paso ---
trozos = []
for duracion, empuje in TRAMOS:
    n_pasos = round(duracion / H)
    trozos.append(np.full(n_pasos, empuje))
us = np.concatenate(trozos)

# --- Simulación con RK4 ---
ts, xs = simular(rk4_step, drone_1d, X0, us, H)

# --- Solución exacta, paso a paso (aceleración constante en cada paso) ---
x_exacta = np.zeros((len(us) + 1, 2))
x_exacta[0] = X0
for k, T in enumerate(us):
    z, v = x_exacta[k]
    a = (T - M * G) / M
    x_exacta[k + 1] = [z + v * H + 0.5 * a * H**2, v + a * H]

# --- Error al final ---
# Con aceleración constante, RK4 reproduce la solución exacta salvo redondeo.
error = np.linalg.norm(xs[-1] - x_exacta[-1])
print(f"Estado final RK4:    z = {xs[-1, 0]:.6f} m, v = {xs[-1, 1]:.6f} m/s")
print(f"Estado final exacto: z = {x_exacta[-1, 0]:.6f} m, v = {x_exacta[-1, 1]:.6f} m/s")
print(f"Error final: {error:.2e}")

# --- Gráficas: empuje, altura y velocidad ---
fig, (ax_u, ax_z, ax_v) = plt.subplots(3, 1, sharex=True, figsize=(8, 9))

# Repetimos el último empuje para dibujar también el último intervalo completo.
ax_u.step(ts, np.append(us, us[-1]), where="post", label="empuje T")
ax_u.axhline(empuje_hover(), ls="--", color="black", label="peso (m·g)")
ax_u.set_ylabel("empuje T [N]")
ax_u.legend()
ax_u.grid(True, alpha=0.2)

ax_z.plot(ts, x_exacta[:, 0], "k-", label="exacta")
ax_z.plot(ts, xs[:, 0], "bo", ms=3, label="RK4")
ax_z.set_ylabel("altura z [m]")
ax_z.legend()
ax_z.grid(True, alpha=0.2)

ax_v.plot(ts, x_exacta[:, 1], "k-", label="exacta")
ax_v.plot(ts, xs[:, 1], "bo", ms=3, label="RK4")
ax_v.set_ylabel("velocidad v [m/s]")
ax_v.set_xlabel("tiempo [s]")
ax_v.legend()
ax_v.grid(True, alpha=0.2)

fig.tight_layout()
carpeta = Path(__file__).parent / "salidas"
carpeta.mkdir(exist_ok=True)
archivo = carpeta / "02_dron_1d_lazo_abierto.png"
fig.savefig(archivo, dpi=200)
print(f"Gráfica guardada en {archivo}")
plt.show()
