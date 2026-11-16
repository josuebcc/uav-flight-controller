import numpy as np

from uav_sim.dynamics.drone_1d import T_MAX, G, M


def control_pd_altura(x, z_ref, kp, kd):
    """Ley PD de altura con feedforward del peso y saturación.

    x     : estado actual [z, v]
    z_ref : altura deseada [m]
    kp, kd: ganancias del PD
    Devuelve el empuje T [N], limitado a [0, T_MAX].
    """
    z, v = x
    e = z_ref - z                      # error de altura
    T = M * G + kp * e - kd * v        # feedforward + P + D
    return np.clip(T, 0, T_MAX)
