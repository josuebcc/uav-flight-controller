def rk4_step(f, x, u, h):
    """Avanza un paso h el sistema ẋ = f(x, u).

    f  : función que recibe (x, u) y devuelve las derivadas
    x  : estado actual, array de NumPy
    u  : entrada constante durante el paso
    h : tamaño del paso
    """
    k1 = f(x, u)
    k2 = f(x + h/2 * k1, u)
    k3 = f(x + h/2 * k2, u)
    k4 = f(x + h * k3, u)

    return x + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)


def euler_step(f, x, u, h):
    """Avanza un paso h con Euler: pendiente al principio del paso."""
    return x + h * f(x, u)


def heun_step(f, x, u, h):
    """Avanza un paso h con Heun: media de la pendiente al principio y al final."""
    k1 = f(x, u)
    x_pred = x + h * k1
    k2 = f(x_pred, u)
    return x + (h / 2) * (k1 + k2)


