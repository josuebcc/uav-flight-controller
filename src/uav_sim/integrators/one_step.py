"""Integradores numéricos de un paso para sistemas ẋ = f(x, u).

Todos tienen la forma general (Chapra, cap. 25)

    x_{k+1} = x_k + h · φ

donde φ estima la pendiente media del estado durante el paso; los métodos
solo se diferencian en cómo calculan φ.

Argumentos comunes a todas las funciones:
- f(x, u): modelo; devuelve las derivadas del estado, no el estado siguiente.
  Es invariante en el tiempo, por eso no recibe t.
- x: estado actual, array de NumPy de cualquier tamaño. Como se opera
  componente a componente, vale igual para una ecuación que para un sistema
  (cada pendiente k es entonces un vector).
- u: entrada, constante durante todo el paso (retenedor de orden cero, ZOH),
  igual que la orden que el controlador digital manda a los motores.
- h: tamaño del paso.
"""


def euler_step(f, x, u, h):
    """Euler: usa la pendiente al principio del paso para todo el paso.

    Orden 1: error local ∝ h², error global ∝ h.
    """
    return x + h * f(x, u)


def heun_step(f, x, u, h):
    """Heun (predictor-corrector, RK2): media de las pendientes al principio y al final.

    Orden 2: error local ∝ h³, error global ∝ h².
    """
    k1 = f(x, u)                    # pendiente al principio
    k2 = f(x + h * k1, u)           # pendiente al final, en el punto predicho por Euler
    return x + (h / 2) * (k1 + k2)  # corrector: se rehace el paso desde x con la media


def rk4_step(f, x, u, h):
    """Runge-Kutta clásico de cuarto orden: media ponderada de cuatro pendientes.

    Orden 4: error local ∝ h⁵, error global ∝ h⁴.
    """
    k1 = f(x, u)               # al principio
    k2 = f(x + h / 2 * k1, u)  # en el centro, llegando con k1
    k3 = f(x + h / 2 * k2, u)  # en el centro otra vez, llegando con k2 (mejor estimación)
    k4 = f(x + h * k3, u)      # al final, llegando con k3 (paso completo)
    return x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # las del centro pesan el doble
