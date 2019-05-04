import TCLab
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

ht = 10 ** -1
hx = 10 ** -1
t0 = 0
x0 = 0
tmax = 0.5
xmax = 1
hx = np.linspace(10 ** -3, 0.1, 10 ** 2 * 2)
u = np.array([TCLab.linear_approximation(
    TCLab.solve_euler_implicit(HX, ht, x0, xmax, t0, tmax), t0, x0, ht, HX, 0.5, 0.5) for HX in hx])
line = np.polyfit(hx, u, 1)
target_line = np.polyval(line, ht)
plt.plot(hx, u, color='green', linestyle=':', marker='.')
plt.plot(hx, target_line, color='blue')
plt.grid(True)
plt.show()

# t = 0.5
# x = 0.5
# num = 100
# colors = ('green', 'red', 'purple')
# for i in range(1, 4):
#     n = np.arange(i, num + 1, 4)
#     func = np.array([TCLab.func(t, x, N) for N in n])
#     func_mean = np.full(func.size, func.mean())
#
#     plt.plot(n, func, color=colors[i - 1], linestyle=':', label=u'period={}'.format(i))
#     plt.plot(n, func_mean, color=colors[i - 1], linestyle='--')
#     print(func_mean[0])
#
# n = np.arange(1, num + 1, 1)
# func = np.array([TCLab.func(t, x, N) for N in n])
# func_mean = np.full(func.size, func.mean())
#
# # plt.plot(n, func, color='yellow')
# plt.plot(n, func_mean, color='blue', linestyle='--')
# print(func_mean[0])
#
# plt.grid(True)
# plt.legend()
# plt.show()

# a = np.linspace(0, 0.5, 10 ** 2)
# num = 150
# # u = np.array([sum([TCLab.func(0, A, n) for n in range(1, num + 1)]) / num for A in a])
# u = np.array([TCLab.func(A, 0) for A in a])
#
# plt.plot(a, u, color='blue')
# plt.grid(True)
# plt.show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# # Make data.
# X = np.arange(0, 1, 0.01)
# T = np.arange(0, 0.5, 0.01)
# X, T = np.meshgrid(X, T)
# num = 11
# Z = TCLab.func(T, X, 100)
# print(Z.shape)
#
# # Plot the surface.
# surf = ax.plot_surface(X, T, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# ax.set_zlim(-0.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()


