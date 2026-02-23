import math

import numpy as np

from kepler.kepler_orbit import KeplerElements, kepler_from_heights, kepler_to_rv
from kepler.constants import MU_EARTH, R_EARTH

import pytest

def test_kepler_from_heights_circular():
    H = 400_000.0
    a, e, T = kepler_from_heights(H, H)

    assert math.isclose(e, 0.0, abs_tol=1e-12)
    r = R_EARTH + H
    expected_T = 2.0 * math.pi * math.sqrt(r ** 3 / MU_EARTH)
    assert math.isclose(T, expected_T, rel_tol=1e-8)


def test_kepler_to_rv_circular_equatorial():
    H = 400_000.0
    elements = KeplerElements(
        nakl_kep=0.0,
        e_kep=0.0,
        H_p_kep=H,
        H_a_kep=H,
        w_kep=0.0,
        teta_kep=0.0,
        Omega_kep=0.0,
    )

    Ri, Vi = kepler_to_rv(elements)

    # Радиус должен быть на оси X, скорость — на оси Y
    r = np.linalg.norm(Ri)
    v = np.linalg.norm(Vi)

    assert Ri[1] == pytest.approx(0.0, abs=1e-6)
    assert Ri[2] == pytest.approx(0.0, abs=1e-6)
    assert Vi[2] == pytest.approx(0.0, abs=1e-6)

    # Проверим соответствие орбитальной скорости и радиуса
    expected_r = R_EARTH + H
    expected_v = math.sqrt(MU_EARTH / expected_r)

    assert r == pytest.approx(expected_r, rel=1e-8)
    assert v == pytest.approx(expected_v, rel=1e-8)

