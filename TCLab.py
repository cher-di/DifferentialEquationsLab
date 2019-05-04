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


def solve_euler_implicit(x: tuple, t: tuple, hx: float, ht: float):
    x0, xmax = x
    t0, tmax = t
    size_x = np.arange(x0, xmax + hx, hx).size
    size_t = np.arange(t0, tmax + ht, ht).size
    size_vertical = size_x + size_t - 2
    size_horizontal = size_x
    u = np.zeros((size_vertical, size_horizontal), dtype=np.float)
    u[:, 0] = 2 * np.linspace(t0, t0 + ht * (size_vertical - 1), size_vertical).T
    u[:, 1] = 3 * np.linspace(t0, t0 + ht * (size_vertical - 1), size_vertical).T ** 2
    u[1, 2:] = ht * (t0 + ht) * np.linspace(x0 + hx * 2, x0 + hx * (size_horizontal - 1), size_horizontal - 2)

    for i in range(1, size_horizontal - 1):
        for j in range(2, size_vertical - i):
            tj = t0 + ht * (j - 1)
            xi = x0 + hx * (i - 1)
            u[j, i + 1] = 2 * u[j, i] - u[j, i - 1] + hx ** 2 * ((u[j + 1, i] - u[j, i]) / ht - tj * np.sinh(xi))

    return u[:size_t, :size_x]


def linear_approximation(u, t0, x0, ht, hx, t, x):
    i = int(np.floor((x - x0) / hx))
    j = int(np.floor((t - t0) / ht))
    return u[j, i]
