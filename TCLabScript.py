import numpy as np
import matplotlib.pyplot as plt
import TCLab

x = (0, 1)
t = (0, 0.5)
hx = 0.01
ht = 0.01

# n = np.arange(1, 501, 1, dtype=np.int)
# u = np.array([TCLab.solve_analytically(0.5, 0.5, i) for i in n])

# for i, val in enumerate(u):
#     if (i + 1) % 20 == 0:
#         print('{}: {}'.format(i + 1, val))

# plt.plot(n, u, color='blue')
# plt.grid(True)
# plt.show()

tt = np.linspace(t[0], t[1], 1000)
for xcurr in np.linspace(0, 1, 5):
    u = TCLab.solve_analytically(tt, xcurr)
    plt.plot(tt, u, label=u'x={}'.format(xcurr))
plt.grid(True)
plt.legend()
plt.show()
