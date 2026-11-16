"""Modelo del dron en 1D: solo movimiento vertical.

Estado x = [z, v]: altura [m] y velocidad vertical [m/s], positivas hacia arriba.
Entrada u = T: empuje total de los motores [N].

Ecuaciones (Newton en vertical):
    ż = v
    v̇ = T/m − g

Hipótesis: masa puntual, sin rozamiento, empuje instantáneo, sin suelo.
"""

import numpy as np

M = 1.0  # masa del dron [kg]
G = 9.81  # gravedad [m/s²]
T_MAX = 2.0 * M * G   # empuje máximo de los motores [N]


def drone_1d(x, u):
    """Derivadas del estado del dron 1D.

    x : estado [z, v]
    u : empuje T [N], constante durante el paso
    Devuelve [ż, v̇].
    """
    z, v = x
    dz_dt = v
    dv_dt = u / M - G
    return np.array([dz_dt, dv_dt])


def empuje_hover():
    """Empuje que compensa el peso y mantiene la velocidad vertical.

    El dron se queda quieto solo si su velocidad inicial es cero.
    """
    return M * G
