import Lab
import matplotlib.pyplot as plt
import numpy as np

x0 = -1
y0 = 4
dy0 = 10
X = (0, 10)
h = 10 ** -2

func = Lab.solve_analytically(x0, y0, dy0)
xa, ya = Lab.prepare_data_for_plotting_func(func, X, h)
xc, yc = Lab.solve_hyung_numerically(x0, y0, dy0, X, h)
xer, yer = xa, np.abs((yc - ya) / ya) * 100

plt.plot(xa, ya, color='green', label=u'Аналитическое решение')
plt.plot(xc, yc, color='blue', label=u'Численное решение')
plt.title(u'Метод Хьюна')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(xer, yer, color='red')
plt.title(u'Невязка')
plt.grid(True)
plt.show()

func = Lab.make_approximation_function(np.array((xc, yc)))
xap = np.arange(X[0], X[1], 10 ** -5)
yap = np.array([func(x) for x in xap])
plt.plot(xap, yap, color='orange')
plt.title(u'Аппроксимация')
plt.grid(True)
plt.show()
