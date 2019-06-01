import numpy as np
import matplotlib.pyplot as plt
import TCLab

x = (0, 1)
t = (0, 0.5)
hx = 0.01
ht = 0.01

# x_test = 0.7
tt = np.linspace(t[0], t[1], int((t[1] - t[0]) / ht) + 1)
u_analytic = np.array([TCLab.solve_analytically(tt, x_test) for x_test in np.arange(x[0], x[1], hx)]).T
u_euler_implicit = TCLab.solve_implicit_schema(x, t, hx, ht)

print('Implicit:', TCLab.get_u_xt(u_euler_implicit, t[0], x[0], ht, hx, 0.5, 0.5))
print('Analytic:', TCLab.get_u_xt(u_analytic, t[0], x[0], ht, hx, 0.5, 0.5))

# plt.plot(tt, u_analytic, label=u'Аналитическое решение')
# plt.plot(tt, u_euler_implicit[:, int(np.floor(x_test / hx))], label=u'Неявная схема')

plt.imshow(u_euler_implicit)
plt.colorbar()

plt.xlabel(u'x')
plt.ylabel(u't')
plt.title(u'Уравнение теплопроводности численное решение')
plt.grid(True)
plt.show()

plt.imshow(u_analytic)
plt.colorbar()

plt.xlabel(u'x')
plt.ylabel(u't')
plt.title(u'Уравнение теплопроводности аналитическое решение')
plt.grid(True)
plt.show()

