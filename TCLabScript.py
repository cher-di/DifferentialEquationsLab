import numpy as np
import matplotlib.pyplot as plt
import TCLab

x = (0, 1)
t = (0, 0.5)
hx = 10 ** -2
ht = 10 ** -2

# n = np.arange(1, 501, 1, dtype=np.int)
# u = np.array([TCLab.solve_analytically(0.5, 0.5, i) for i in n])

# for i, val in enumerate(u):
#     if (i + 1) % 20 == 0:
#         print('{}: {}'.format(i + 1, val))

# plt.plot(n, u, color='blue')
# plt.grid(True)
# plt.show()

# tt = np.linspace(t[0], t[1], 1000)
# for xcurr in np.linspace(0, 1, 5):
#     u = TCLab.solve_analytically(tt, xcurr)
#     plt.plot(tt, u, label=u'x={}'.format(xcurr))
# plt.grid(True)
# plt.legend()
# plt.show()

x_test = 0.4
tt = np.linspace(t[0], t[1], int((t[1] - t[0]) / ht) + 1)
u_analytic = TCLab.solve_analytically(tt, x_test)
u_euler_explicit = TCLab.solve_euler_explicit(x, t, hx, ht)
u_euler_implicit = TCLab.solve_euler_implicit(x, t, hx, ht)

plt.plot(tt, u_analytic, label=u'Аналитическое решение')
plt.plot(tt, u_euler_explicit[:, int(np.floor(x_test / hx))], label=u'Явная схема')
plt.plot(tt, u_euler_implicit[:, int(np.floor(x_test / hx))], label=u'Неявная схема')

plt.xlabel(u't')
plt.ylabel(u'u')
plt.title(u'x = {}'.format(x_test))
plt.grid(True)
plt.legend()
plt.show()
