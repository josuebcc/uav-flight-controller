import numpy as np

def rk4_step(f, x, u, dt):
    """Avanza un paso dt el sistema ẋ = f(x, u).

    f  : función que recibe (x, u) y devuelve las derivadas
    x  : estado actual, array de NumPy
    u  : entrada (el empuje), constante durante el paso
    dt : tamaño del paso
    """