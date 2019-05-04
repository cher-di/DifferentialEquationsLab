import math
import numpy as np


def func(t: float, x: float, n=10):
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
    if not (0 <= x <= 1 and 0 <= t <= 0.5):
        raise ArithmeticError

    U = (3 * t ** 2 - 2 * t) * x + 2 * t

    a = lambda n: 2 * (-1) ** (n + 1) * math.pi * n * math.sinh(1) / (1 + (math.pi * n) ** 2) + \
                  12 * (-1) ** n / math.pi / n * (1 - 1 / math.pi / n) + 12 / (math.pi * n) ** 2
    b = lambda n: 4 / (math.pi * n) ** 2 * ((-1) ** n - 1) - 4 / math.pi / n
    Cn = lambda n: a(n) * t + (b(n) - a(n) / (math.pi * n) ** 2) * \
                   (1 - math.exp(-(math.pi * n) ** 2 * t) / (math.pi * n) ** 2)
    v = 0
    for i in range(1, n + 1):
        v += Cn(n) * math.sin(math.pi * n * x)

    return U + v


def func1(t: float, x: float, n=10):
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
    if not (0 <= x <= 1 and 0 <= t <= 0.5):
        raise ArithmeticError

    U = x + 1

    Cn = lambda n: (6 * (-1) ** (n + 1) + 2) / (math.pi * n) ** 3 * (1 - math.exp(-(math.pi * n) ** 2 * t))
    v = 0
    for i in range(1, n + 1):
        v += Cn(n) * math.sin(math.pi * n * x)

    return U + v


def solve_euler_implicit(hx: float, ht: float, x0: float, xmax: float, t0: float, tmax: float):
    # начальные условия
    # ut0 = 2 * np.arange(mint, maxt + ht, ht)
    # ut1 = 3 * np.arange(mint, maxt + ht, ht) ** 2
    # u0x = np.zeros(np.arange(minx, maxx + hx, hx).size)
    #
    # grid = np.zeros((ut0.size, u0x.size))
    # grid[:, 0] = ut0.T
    # grid[:, 1] = ut1.T
    # grid[0, :] = u0x
    #
    # x = np.arange(minx, maxx + hx, hx)
    # t = np.arange(mint, maxt + ht, ht)

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
            u[j, i + 1] = 2 * u[j, i] - u[j, i - 1] + hx ** 2 * ((u[j + 1, i] - u[j, i]) / ht - tj * math.sinh(xi))

    # for j in range(1, size_t):
    #     for i in range(2, size_x + size_t - 1 - j):
    #         tj = t0 + ht * j
    #         xi = x0 + hx * i
    #         u[j, i] = u[j - 1, i] + ht * ((u[j - 1, i + 1] - 2 * u[j - 1, i] + u[j - 1, i - 1]) / hx ** 2 +
    #                                       tj * np.sinh(xi))

    return u[:size_t, :size_x]


def linear_approximation(u, t0, x0, ht, hx, t, x):
    i = int(math.floor((x - x0) / hx))
    j = int(math.floor((t - t0) / ht))
    return u[j, i]
