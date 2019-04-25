import math
import numpy as np


def solve_analytically(x0: float, y0: float, dy0: float):
    """
    Принимает на вход условия для задачи Коши и возврщает функцию, из семейства функций,
    удовлетворяющую условиям задачи Коши
    Уравнение: y'' - 6*y' + 9*y = x^2 - x + 3
    Общее решение: y = C1*e^3x + C2*xe^3x + 1/9*x^2 + 1/27*x + 1/3

    :param x0: точка, для которой известны начальные условия задачи Коши
    :param y0: функция в точке x0
    :param dy0: производная функции в точке x0
    :return: функция, удовлетворяющая заданным условиям
    """
    A1 = [math.exp(3 * x0), x0 * math.exp(3 * x0)]
    B1 = y0 - (1 / 9 * x0 ** 2 + 1 / 27 * x0 + 1 / 3)
    A2 = [3 * math.exp(3 * x0), (3 * x0 + 1) * math.exp(3 * x0)]
    B2 = dy0 - (2 / 9 * x0 + 1 / 27)
    A = np.array([A1,
                  A2])
    B = np.array([[B1],
                  [B2]])
    X = np.linalg.solve(A, B)
    [C1], [C2] = X.tolist()

    return lambda x: C1 * math.exp(3 * x) + C2 * x * math.exp(3 * x) + 1 / 9 * x ** 2 + 1 / 27 * x + 1 / 3


def prepare_data_for_plotting_func(func: callable, borders: tuple, step=0.01):
    """
    Подготавливает данные для построения графика функции

    :param func: сама фукнция
    :param borders: границы
    :param step: шаг дискретезации
    :return: (x, y)
    """
    start, finish = borders
    x = np.arange(start, finish + step, step)
    y = np.array([func(el) for el in x])
    return x, y


def solve_euler_numerically(x0: float, y0: float, dy0: float, borders: tuple, h=0.01):
    """
    Принимает на вход условия для задачи Коши и границы отрезка, на котором нужно решить ДУ,
    решает уравнение методом Эйлера с помощью построения сетки на необходимом отрезке
    и возврщает решение в узлах сетки.
    Уравнение: y'' - 6*y' + 9*y = x^2 - x + 3

    :param x0: точка, для которой известны начальные условия задачи Коши
    :param y0: функция в точке x0
    :param dy0: производная функции в точке x0
    :param borders: границы
    :param h: шаг сетки
    :return: двумерный массив, первая строка - X, вторая - Y
    """
    left, right = borders
    if x0 > left:
        raise ArithmeticError

    xj, dyj, yj = x0, dy0, y0

    if x0 < left:
        for xj in np.arange(x0, left + h, h):
            dyj_next = dyj + h * (6 * dyj - 9 * yj + xj ** 2 - xj + 3)
            yj_next = yj + h * dyj
            dyj = dyj_next
            yj = yj_next

    steps_num = np.arange(left, right + h, h).size
    shape = (3, steps_num)
    grid = np.empty(shape, np.float)
    grid[:, 0] = [xj,
                  dyj,
                  yj]
    grid[0] = np.arange(left, right + h, h)
    for j in np.arange(steps_num - 1):
        xj, dyj, yj = grid[:, j]
        grid[1, j + 1] = dyj + h * (6 * dyj - 9 * yj + xj ** 2 - xj + 3)
        grid[2, j + 1] = yj + h * dyj

    return grid[(0, 2), :]


def solve_hyung_numerically(x0: float, y0: float, dy0: float, borders: tuple, h=0.01):
    """
    Принимает на вход условия для задачи Коши и границы отрезка, на котором нужно решить ДУ,
    решает уравнение методом Хьюна с помощью построения сетки на необходимом отрезке
    и возврщает решение в узлах сетки.
    Уравнение: y'' - 6*y' + 9*y = x^2 - x + 3

    :param x0: точка, для которой известны начальные условия задачи Коши
    :param y0: функция в точке x0
    :param dy0: производная функции в точке x0
    :param borders: границы
    :param h: шаг сетки
    :return: двумерный массив, первая строка - X, вторая - Y
    """
    left, right = borders
    if x0 > left:
        raise ArithmeticError

    xj, dyj, yj = x0, dy0, y0

    if x0 < left:
        for xj in np.arange(x0, left + h, h):
            xj_temp = xj + h
            dyj_temp = dyj + h * (6 * dyj - 9 * yj + xj ** 2 - xj + 3)
            yj_temp = yj + h * dyj
            dyj_next = dyj + h/2*(6 * dyj - 9 * yj + xj ** 2 - xj + 3 +
                                  6 * dyj_temp - 9 * yj_temp + xj_temp ** 2 - xj_temp + 3)
            yj_next = yj + h/2*(dyj + dyj_temp)
            dyj = dyj_next
            yj = yj_next

    steps_num = np.arange(left, right + h, h).size
    shape = (3, steps_num)
    grid = np.empty(shape, np.float)
    grid[:, 0] = [xj,
                  dyj,
                  yj]
    grid[0] = np.arange(left, right + h, h)
    for j in np.arange(steps_num - 1):
        xj, dyj, yj = grid[:, j]

        xj_temp = xj + h
        dyj_temp = dyj + h * (6 * dyj - 9 * yj + xj ** 2 - xj + 3)
        yj_temp = yj + h * dyj
        dyj_next = dyj + h / 2 * (6 * dyj - 9 * yj + xj ** 2 - xj + 3 +
                                  6 * dyj_temp - 9 * yj_temp + xj_temp ** 2 - xj_temp + 3)
        yj_next = yj + h / 2 * (dyj + dyj_temp)

        grid[1, j + 1] = dyj_next
        grid[2, j + 1] = yj_next

    return grid[(0, 2), :]
