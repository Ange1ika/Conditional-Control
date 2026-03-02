from __future__ import annotations

import numpy as np


B0 = 398600.4418e9      # м^3/с^2  (μ)
B2 = 0.1755650e26       # м^5/с^2
B4 = 1.563955e36        # м^7/с^2


def gravity_normal_earth(x, y, z, *, b0=B0, b2=B2, b4=B4):
    """
    Модель гравитационного поля нормальной Земли с учётом 2-й и 4-й зональных гармоник.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    r2 = x*x + y*y + z*z
    if np.any(r2 == 0.0):
        raise ValueError("Нельзя вычислить поле в точке r=0 (x=y=z=0).")

    r = np.sqrt(r2)

    inv_r = 1.0 / r
    inv_r3 = inv_r**3
    inv_r4 = inv_r**4
    inv_r6 = inv_r**6

    q = z * inv_r

    dg0 = 3.0 * b2 * q * inv_r4 + 2.5 * b4 * (7.0*q**3 - 3.0*q) * inv_r6
    dgr = -1.5 * b2 * (5.0*q**2 - 1.0) * inv_r4 - 1.875 * b4 * (21.0*q**4 - 14.0*q**2 + 1.0) * inv_r6

    gx = -b0 * x * inv_r3 + dgr * x * inv_r
    gy = -b0 * y * inv_r3 + dgr * y * inv_r
    gz = -b0 * z * inv_r3 + dgr * z * inv_r + dg0

    return gx, gy, gz