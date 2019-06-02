import numpy as np
import matplotlib.pyplot as plt
import TCLab

x = (0, 1)
t = (0, 0.5)
hx = 0.01
ht = 0.01

# x_test = 0.7
tt = np.arange(t[0], t[1] + ht, ht)
u_analytic = np.array([TCLab.solve_analytically(tt, x_test) for x_test in np.arange(x[0], x[1] + hx, hx)]).T
u_euler_implicit = TCLab.solve_implicit_schema(x, t, hx, ht)

print('Implicit:', TCLab.get_u_xt(u_euler_implicit, x, t, 0.5, 0.5))
print('Analytic:', TCLab.get_u_xt(u_analytic, x, t, 0.5, 0.5))

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

