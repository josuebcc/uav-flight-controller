"""Bucle de simulación con entrada constante en cada paso (ZOH)."""

import numpy as np


def simular(solver, f, x0, us, h):
    """Simula el sistema ẋ = f(x, u) aplicando la entrada us[k] durante el paso k.

    solver : integrador (euler_step, heun_step o rk4_step)
    f      : modelo, f(x, u) -> derivadas del estado
    x0     : estado inicial (array)
    us     : entradas, una por paso (la entrada k se mantiene constante en el paso k)
    h      : tamaño del paso

    Devuelve (ts, xs):
    ts : instantes t_k = k·h, de 0 a N·h  (N + 1 valores)
    xs : estados en esos instantes, una fila por instante  (N + 1 filas)
    """
    N = len(us)
    ts = np.arange(N + 1) * h
    xs = np.zeros((N + 1, len(x0)))
    xs[0] = x0

    for k in range(N):
        xs[k + 1] = solver(f, xs[k], us[k], h)

    return ts, xs


def simular_lazo_cerrado(solver, f, control, x0, N, h, **parametros_control):
    """Simula N pasos en lazo cerrado: u = control(x, **parametros_control).

    solver  : integrador (p. ej. rk4_step)
    f       : modelo, f(x, u)
    control : función que recibe el estado x y sus parámetros y devuelve u
    x0      : estado inicial
    N       : número de pasos
    h       : paso de integración [s]
    parametros_control : parámetros adicionales del controlador, pasados por nombre
    Devuelve ts, xs y us (el u aplicado en cada paso, para dibujarlo).
    """
    ts = np.arange(N + 1) * h
    xs = np.zeros((N + 1, len(x0)))
    us = np.zeros(N)
    xs[0] = x0
    for k in range(N):
        us[k] = control(xs[k], **parametros_control)  # 1. calcula el empuje con el estado actual
        xs[k + 1] = solver(f, xs[k], us[k], h)    # 2. un paso con ese u, mantenido durante h
    return ts, xs, us
