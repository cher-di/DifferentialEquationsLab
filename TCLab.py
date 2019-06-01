import numpy as np


def solve_analytically(t, x, n=500):
    """
    Возврщает значение функции теплопроводности, заданной ДУ:
    du/dt = d2u / dx2 + t*sh(x)
    С начальными условиями:
    u(0, x) = 0
    u(t, 0) = 2*t
    u(t, 1) = 3*t^2
    В области (t, x) c [0; 0.5] x [0; 1]

    :param t:
    :param x:
    :param n: количество членов в разложении
    :return: значение функции в точке (t, x)
    """
    U = (3 * t ** 2 - 2 * t) * x + 2 * t

    a = lambda n: 2 * (-1) ** (n + 1) * np.pi * n * np.sinh(1) / (1 + (np.pi * n) ** 2) + \
                  12 * (-1) ** n / np.pi / n * (1 - 1 / np.pi / n) + 12 / (np.pi * n) ** 2
    b = lambda n: 4 / (np.pi * n) ** 2 * ((-1) ** n - 1) - 4 / np.pi / n
    Cn = lambda n: a(n) * t + (b(n) - a(n) / (np.pi * n) ** 2) * \
                   (1 - np.exp(-(np.pi * n) ** 2 * t) / (np.pi * n) ** 2)
    v = 0
    for i in range(1, n + 1):
        v += Cn(i) * np.sin(np.pi * i * x)

    return U + v


def get_u_xt(u, t0, x0, ht, hx, t, x):
    i = int(np.floor((x - x0) / hx))
    j = int(np.floor((t - t0) / ht))
    return u[j, i]


def solve_implicit_schema(x: tuple, t: tuple, hx: float, ht: float):
    x0, xmax = x
    t0, tmax = t

    nmax, kmax = int((tmax - t0) / ht) + 1, int((xmax - x0) / hx) + 1

    u = np.zeros((nmax, kmax))
    u[0, :] = 0
    u[:, 0] = np.linspace(t0, tmax, nmax).T * 2
    u[:, -1] = np.linspace(t0, tmax, nmax).T ** 2 * 3

    f = lambda k, n: (t0 + (k-1) * ht) * np.sinh(x0 + (k-1) * hx)

    for n in range(1, nmax):
        A = np.eye(kmax - 2, k=-1) / hx ** 2
        B = np.eye(kmax - 2) * (-2 / hx ** 2 - 1 / ht)
        C = np.eye(kmax - 2, k=1) / hx ** 2
        equations_koef = A + B + C

        F = np.zeros(kmax - 2)
        F[0] = -u[n - 1, 1] / ht - f(1, n) - u[n, 0] / hx ** 2
        F[kmax - 3] = -u[n - 1, kmax - 2] / ht - f(kmax - 2, n) - u[n, kmax - 1] / hx ** 2
        for k in range(1, kmax - 3):
            F[k] = -u[n - 1, k + 1] / ht - f(k + 1, n)
        ucurr = np.linalg.solve(equations_koef, F.T)
        u[n, 1:-1] = ucurr.T
    return u
