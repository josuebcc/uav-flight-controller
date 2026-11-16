import numpy as np
import pytest

from uav_sim.dynamics.drone_1d import G, drone_1d, empuje_hover
from uav_sim.integrators.one_step import rk4_step


def integrar(solver, f, x0, u, T, h):
    """Integra desde t=0 hasta T con pasos h y devuelve el estado final."""
    n = round(T / h)
    x = x0
    for _ in range(n):
        x = solver(f, x, u, h)
    return x


def test_caida_libre():
    """Sin empuje: z = z0 − ½·g·t² y v = −g·t."""
    z0, v0, T = 100.0, 0.0, 3.0
    x = integrar(rk4_step, drone_1d, np.array([z0, v0]), 0.0, T, 0.01)

    assert x[0] == pytest.approx(z0 + v0 * T - 0.5 * G * T**2, rel=1e-12)
    assert x[1] == pytest.approx(v0 - G * T, rel=1e-12)


def test_hover():
    """Con T = m·g, el dron parte quieto y no se mueve."""
    x0 = np.array([5.0, 0.0])
    x = integrar(rk4_step, drone_1d, x0, empuje_hover(), 2.0, 0.01)

    assert x[0] == pytest.approx(5.0, abs=1e-12)
    assert x[1] == pytest.approx(0.0, abs=1e-12)
