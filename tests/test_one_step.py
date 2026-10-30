import numpy as np
import pytest

from uav_sim.integrators.one_step import euler_step, heun_step, rk4_step


def integrar(tipo, f, x0, u, T, h):
    """Integra la ecuación desde t=0 y devuelve el estado final."""
    # tipo: integrador que usamos (Euler, Heun o RK4).
    # f: función que devuelve las derivadas del estado.
    # x0: estado inicial.
    # u: entrada constante; en estos ejemplos no se utiliza.
    # T: tiempo total de simulación.
    # h: tamaño de cada paso.
    # En estos tests, T es un múltiplo de h.
    n = round(T / h)

    x = x0
    for _ in range(n):
        # Calculamos el siguiente estado a partir del actual.
        x = tipo(f, x, u, h)

    return x


def exponencial(x, u):
    """Ecuación dx/dt = x. Con x(0)=1, la solución es x(t)=exp(t)."""
    # Devuelve la derivada, no el siguiente estado.
    dx_dt = x
    return dx_dt


def oscilador(x, u):
    """Oscilador: derivada de la posición = velocidad; aceleración = -posición."""
    # El estado contiene dos componentes: posición y velocidad.
    posicion, velocidad = x

    dposicion_dt = velocidad
    dvelocidad_dt = -posicion

    # Las derivadas deben mantener el mismo orden que el estado.
    return np.array([dposicion_dt, dvelocidad_dt])


# Pytest ejecuta test_orden tres veces, una por cada pareja:
# tipo=euler_step, orden=1
# tipo=heun_step,  orden=2
# tipo=rk4_step,   orden=4
@pytest.mark.parametrize(
    ("tipo", "orden"),
    [
        (euler_step, 1),
        (heun_step, 2),
        (rk4_step, 4),
    ],
)
def test_orden(tipo, orden):
    """Comprueba cuánto disminuye el error al reducir el paso a la mitad."""
    x0 = np.array([1.0])

    # Solución exacta de dx/dt = x en t=1, empezando en x=1.
    exacto = np.exp(1.0)

    # Calculamos el estado final con dos tamaños de paso.
    # [0] extrae el único componente del array devuelto.
    resultado_h = integrar(tipo, exponencial, x0, 0.0, 1.0, 0.1)[0]
    resultado_h2 = integrar(tipo, exponencial, x0, 0.0, 1.0, 0.05)[0]

    # Error absoluto respecto a la solución exacta.
    error_h = abs(resultado_h - exacto)
    error_h2 = abs(resultado_h2 - exacto)

    # Al dividir h entre 2, esperamos que el error disminuya aproximadamente:
    # Euler: 2 veces; Heun: 4 veces; RK4: 16 veces.
    # Admitimos una tolerancia relativa del 15 % en ese cociente.
    assert error_h / error_h2 == pytest.approx(2**orden, rel=0.15)


def test_rk4_oscilador():
    """Comprueba que RK4 calcula correctamente posición y velocidad."""
    T = 2.0

    # Posición inicial = 1; velocidad inicial = 0.
    # Integramos hasta t=2 con pasos de 0.01: 200 pasos.
    x = integrar(
        rk4_step,
        oscilador,
        np.array([1.0, 0.0]),
        0.0,
        T,
        0.01,
    )

    # Comprobamos que el resultado sea un array de dos componentes.
    assert x.shape == (2,)

    # Para estas condiciones iniciales, la solución exacta es:
    # posición(t) = cos(t), velocidad(t) = -sin(t).
    # Permitimos un error absoluto máximo de 1e-8 en cada componente.
    assert x[0] == pytest.approx(np.cos(T), abs=1e-8)
    assert x[1] == pytest.approx(-np.sin(T), abs=1e-8)