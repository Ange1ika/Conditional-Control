"""
Функции для работы с кеплеровыми элементами орбиты.

Основная цель для лабораторной:
- реализовать «перевод Кеплера» — получить радиус‑вектор и скорость в ГСК
  по набору орбитальных элементов.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from constants import MU_EARTH, R_EARTH


@dataclass
class KeplerElements:
    """
    Кеплеровы элементы орбиты в форме, максимально близкой к ipo.h.

    Все углы в радианах, высоты в метрах.
    """

    nakl_kep: float   # наклонение i
    e_kep: float      # эксцентриситет e
    H_p_kep: float    # высота перицентра над поверхностью Земли
    H_a_kep: float    # высота апоцентра над поверхностью Земли
    w_kep: float      # аргумент перицентра omega
    teta_kep: float   # истинная аномалия ν
    Omega_kep: float  # долгота восходящего узла omega (большое)


def kepler_from_heights(
    H_p_kep: float,
    H_a_kep: float,
    mu: float = MU_EARTH,
    R: float = R_EARTH,
) -> Tuple[float, float, float]:
    """
    По высотам перицентра и апоцентра над Землёй вычислить:
      - большую полуось a,
      - эксцентриситет e,
      - период обращения T.

    H_p_kep, H_a_kep — в метрах.
    Возвращает (a, e, T).
    """
    r_p = R + H_p_kep
    r_a = R + H_a_kep

    a = 0.5 * (r_p + r_a)
    e = (r_a - r_p) / (r_a + r_p)
    T = 2.0 * np.pi * np.sqrt(a ** 3 / mu)

    return a, e, T


def solve_kepler(
    M: float,
    e: float,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> Tuple[float, float]:
    """
    Решение кеплерова уравнения:
        M = E - e * sin(E)
    методом Ньютона.

    На вход:
      - M: средняя аномалия (рад),
      - e: эксцентриситет.
    Возвращает:
      - E: эксцентрическая аномалия (рад),
      - nu: истинная аномалия (рад).
    """
    M = np.mod(M, 2.0 * np.pi)
    E = M if e < 0.8 else np.pi  # начальное приближение

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1.0 - e * np.cos(E)
        delta = f / f_prime
        E -= delta
        if abs(delta) < tol:
            break

    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0),
    )
    return float(E), float(nu)


def kepler_to_rv(
    elements: KeplerElements,
    mu: float = MU_EARTH,
    R: float = R_EARTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Перевод кеплеровых элементов (как в ipo.h) в радиус‑вектор и скорость в ГСК.

    Возвращает (Ri, Vi) — numpy‑векторы длины 3 (м, м/с).
    """
    # Восстанавливаем полуось a и эксцентриситет из высот
    a, e_from_heights, T = kepler_from_heights(
        elements.H_p_kep,
        elements.H_a_kep,
        mu=mu,
        R=R,
    )

    # Экцентриситет берём из элементов, но можем свериться с расчётным
    e = elements.e_kep

    # Истинная аномалия
    nu = elements.teta_kep

    # Радиус в перефокальной системе
    r = a * (1.0 - e ** 2) / (1.0 + e * np.cos(nu))

    r_pf = np.array(
        [
            r * np.cos(nu),
            r * np.sin(nu),
            0.0,
        ]
    )

    # Скорость в перефокальной системе
    h = np.sqrt(mu * a * (1.0 - e ** 2))
    v_pf = np.array(
        [
            -mu / h * np.sin(nu),
            mu / h * (e + np.cos(nu)),
            0.0,
        ]
    )

    # Углы ориентации орбиты
    i = elements.nakl_kep
    Omega = elements.Omega_kep
    w = elements.w_kep

    cos_O = np.cos(Omega)
    sin_O = np.sin(Omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_w = np.cos(w)
    sin_w = np.sin(w)

    # Матрицы поворота
    R3_O = np.array(
        [
            [cos_O, -sin_O, 0.0],
            [sin_O, cos_O, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    R1_i = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_i, -sin_i],
            [0.0, sin_i, cos_i],
        ]
    )

    R3_w = np.array(
        [
            [cos_w, -sin_w, 0.0],
            [sin_w, cos_w, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Переход из перефокальной системы в инерциальную (ГСК)
    Q = R3_O @ R1_i @ R3_w

    Ri = Q @ r_pf
    Vi = Q @ v_pf

    return Ri, Vi

