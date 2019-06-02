import numpy as np
import matplotlib.pyplot as plt
import TCLab

x = (0, 1)
t = (0, 0.5)
hx = 0.01
ht = 0.01

u_euler_implicit = TCLab.solve_implicit_schema(x, t, hx, ht)

print('Implicit:', TCLab.get_u_xt(u_euler_implicit, x, t, 0.5, 0.5))

plt.imshow(u_euler_implicit)
plt.colorbar()

plt.xlabel(u'x')
plt.ylabel(u't')
plt.title(u'Уравнение теплопроводности численное решение')
plt.grid(True)
plt.show()

step = np.array((0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001))
u_comparison = np.zeros((step.size, step.size))
for i, hx in enumerate(step):
    for j, ht in enumerate(step):
        u = TCLab.solve_implicit_schema(x, t, hx, ht)
        u_comparison[i, j] = TCLab.get_u_xt(u, x, t, 0.5, 0.5)
        print('Completed ({}, {})'.format(hx, ht))

plt.imshow(u_comparison)
plt.colorbar()

plt.xlabel(u'hx')
plt.ylabel(u'ht')
plt.title(u'Зависимость значения u(0.5, 0.5) от шага сетки')
plt.grid(True)
plt.show()

plt.plot(1/step, u_comparison[:, 2], color='green', marker='.')
plt.xlabel(u'hx')
plt.ylabel(u'u(0.5, 0.5)')
plt.title(u'Зависимость значения u(0.5, 0.5) от шага сетки по координате х')
plt.grid(True)
plt.show()

plt.plot(1/step, u_comparison[2, :], color='darkblue', marker='.')
plt.xlabel(u'ht')
plt.ylabel(u'u(0.5, 0.5)')
plt.title(u'Зависимость значения u(0.5, 0.5) от шага сетки по координате t')
plt.grid(True)
plt.show()
