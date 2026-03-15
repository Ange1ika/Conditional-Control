import numpy as np

def orbital_elements_from_state(r, v, mu=398600.4418e9): # гравитационный параметр Земли в м^3/с^2
    """
    r : numpy array (3,)   радиус-вектор [m]
    v : numpy array (3,)   скорость [m/s]
    mu: float              гравитационный параметр

    возвращает:
    a, e, i, Omega, omega, theta
    """

    r = np.array(r)
    v = np.array(v)

    # 1. расстояние
    r_norm = np.linalg.norm(r)

    # 2. скорость
    v_norm = np.linalg.norm(v)

    # 3. радиальная скорость
    vr = np.dot(r, v) / r_norm

    # 4. угловой момент
    h_vec = np.cross(r, v)

    # 5. модуль углового момента
    h = np.linalg.norm(h_vec)

    # 6. наклонение
    i = np.arccos(h_vec[2] / h)

    # 7. вектор линии узлов
    K = np.array([0, 0, 1])
    N = np.cross(K, h_vec)

    # 8. модуль N
    N_norm = np.linalg.norm(N)

    # 9. долгота восходящего узла
    if N_norm != 0:
        Omega = np.arccos(N[0] / N_norm)
        if N[1] < 0:
            Omega = 2*np.pi - Omega
    else:
        Omega = 0

    # 10. вектор эксцентриситета
    e_vec = (1/mu)*((v_norm**2 - mu/r_norm)*r - r_norm*vr*v)

    # 11. эксцентриситет
    e = np.linalg.norm(e_vec)

    # 12. аргумент перицентра
    if N_norm != 0 and e > 1e-10:
        omega = np.arccos(np.dot(N, e_vec)/(N_norm*e))
        if e_vec[2] < 0:
            omega = 2*np.pi - omega
    else:
        omega = 0

    # 13. истинная аномалия
    if e > 1e-10:
        theta = np.arccos(np.dot(e_vec, r)/(e*r_norm))
        if vr < 0:
            theta = 2*np.pi - theta
    else:
        theta = 0

    # большая полуось
    a = 1/(2/r_norm - v_norm**2/mu)

    return {
        "h": h, # модуль углового момента
        "i": i, # наклонение
        "Omega": Omega, # долгота восходящего узла
        "e": e, # эксцентриситет
        "omega": omega, # аргумент перицентра
        "theta": theta # истинная аномалия
    }
