"""
Пример использования функций перевода Кеплера.

Скрипт создаёт простые кеплеровы элементы и выводит
полученные радиус‑вектор и скорость.
"""

from __future__ import annotations

import math

import numpy as np

from gravity import gravity_normal_earth
from kepler_orbit import KeplerElements, kepler_to_rv


def demo() -> None:
    # Пример: круговая орбита на высоте 400 км, без наклонения и поворотов
    H = 400_000.0  # м
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

    print("Пример для круговой орбиты на высоте 400 км")
    print("Ri (м):", Ri)
    print("Vi (м/с):", Vi)


if __name__ == "__main__":
    # demo()

    g = gravity_normal_earth(6400e3, 0, 0)
    print(g)
    print(np.linalg.norm(g))

