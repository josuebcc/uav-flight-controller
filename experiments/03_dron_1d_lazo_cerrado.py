"""Experimento 03: respuesta de altura con control PD al variar Kp y Kd."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uav_sim.control.pd_altura_dron1d import control_pd_altura
from uav_sim.dynamics.drone_1d import drone_1d
from uav_sim.integrators.one_step import rk4_step
from uav_sim.simulacion import simular_lazo_cerrado

# --- Parámetros ---
H = 0.01            # paso [s]
T_FINAL = 6.0       # duración [s]
N = int(T_FINAL / H)  # número de pasos
X0 = np.array([0.0, 0.0])
Z_REF = 1.0

CASOS_KP = [1.0, 2.0, 4.0, 8.0]
CASOS_KD = [0.0, 2.0, 4.0, 10.0]

KP_FIJO = 4.0
KD_FIJO = 4.0

fig, axes = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(12, 12))

for fila in range(4):
    # Columna izquierda: cambia Kp y mantiene Kd fijo.
    kp = CASOS_KP[fila]
    ts, xs, us = simular_lazo_cerrado(
        rk4_step, drone_1d, control_pd_altura, X0, N, H,
        z_ref=Z_REF, kp=kp, kd=KD_FIJO,
    )

    z = xs[:, 0]
    overshoot = max(0.0, (np.max(z) - Z_REF) / abs(Z_REF - X0[0]) * 100)
    print(f"Kp = {kp:.1f}, Kd = {KD_FIJO:.1f}: overshoot = {overshoot:.2f} %")

    axes[fila, 0].plot(ts, z)
    axes[fila, 0].set_title(
        f"Kp = {kp}, Kd = {KD_FIJO} | overshoot = {overshoot:.1f} %"
    )

    # Columna derecha: cambia Kd y mantiene Kp fijo.
    kd = CASOS_KD[fila]
    ts, xs, us = simular_lazo_cerrado(
        rk4_step, drone_1d, control_pd_altura, X0, N, H,
        z_ref=Z_REF, kp=KP_FIJO, kd=kd,
    )

    z = xs[:, 0]
    overshoot = max(0.0, (np.max(z) - Z_REF) / abs(Z_REF - X0[0]) * 100)
    print(f"Kp = {KP_FIJO:.1f}, Kd = {kd:.1f}: overshoot = {overshoot:.2f} %")

    axes[fila, 1].plot(ts, z)
    axes[fila, 1].set_title(
        f"Kp = {KP_FIJO}, Kd = {kd} | overshoot = {overshoot:.1f} %"
    )

    for columna in range(2):
        ax = axes[fila, columna]
        ax.axhline(Z_REF, color="k", ls="--", label="referencia")
        ax.set_ylabel("altura [m]")
        ax.grid(True, alpha=0.2)

axes[-1, 0].set_xlabel("tiempo [s]")
axes[-1, 1].set_xlabel("tiempo [s]")

fig.tight_layout()
carpeta = Path(__file__).parent / "salidas"
carpeta.mkdir(exist_ok=True)
archivo = carpeta / "03_dron_1d_lazo_cerrado.png"
fig.savefig(archivo, dpi=200)
print(f"Gráfica guardada en {archivo}")
plt.show()
