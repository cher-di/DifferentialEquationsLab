import math
import numpy as np


def solve_cauchy(x0: float, y0: float, dy0: float):
    """
    Принимает на вход условия для задачи Коши и возврщает функцию, из семейства функций,
    удовлетворяющую условиям задачи Коши
    Семейство функций имеет вид: y = C1*e^3x + C2*xe^3x + 1/9*x^2 + 1/27*x + 1/3

    :param x0: точка, для которой известны начальные условия задачи Коши
    :param y0: функция в точке x0
    :param dy0: производная функции в точке x0
    :return: функция, удовлетворяющая заданным условиям
    """
    A1 = [math.exp(3*x0), x0*math.exp(3*x0)]
    B1 = y0 - (1/9*x0**2 + 1/27*x0 + 1/3)
    A2 = [3*math.exp(3*x0), (3*x0 + 1)*math.exp(3*x0)]
    B2 = dy0 - (2/9*x0 + 1/27)
    A = np.array([A1,
                  A2])
    B = np.array([[B1],
                  [B2]])
    X = np.linalg.solve(A, B)
    [C1], [C2] = X.tolist()

    return lambda x: C1 * math.exp(3 * x) + C2 * x * math.exp(3 * x) + 1 / 9 * x ** 2 + 1 / 27 * x + 1 / 3


def prepare_data_for_plotting_func(func: callable, X: tuple, step=0.01):
    """
    Подготавливает данные для построения графика функции

    :param func: сама фукнция
    :param X: Границы
    :param step: шаг дискретезации
    :return: (x, y)
    """
    start, finish = X
    x = np.arange(start, finish, step)
    y = np.array([func(el) for el in x])
    return x, y
