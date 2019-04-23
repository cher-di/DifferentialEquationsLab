import Lab
import matplotlib.pyplot as plt

x0 = 5
y0 = 4
dy0 = 2
X = (0, 10)

func = Lab.solve_cauchy(x0, y0, dy0)
x, y = Lab.prepare_data_for_plotting_func(func, X)

plt.plot(x, y, color='green', label=u'x0={}; y0={}; dy0={}'.format(x0, y0, dy0))
plt.title(u'Решение задачи Коши')
plt.legend()
plt.grid(True)
plt.show()
