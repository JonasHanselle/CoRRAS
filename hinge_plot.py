from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

plt.locator_params(axis="x", numticks=4)
plt.locator_params(axis="y", numticks=4)
plt.locator_params(axis="z", numticks=4)

fig = plt.figure(figsize=plt.figaspect(0.8))
ax = fig.add_subplot(111, projection='3d')

y_hat_i = np.arange(-2, 2, 0.09)
y_hat_j = np.arange(-2, 2, 0.09)
X, Y = np.meshgrid(y_hat_i, y_hat_j)
epsilon = 1
Z = np.maximum(0, epsilon - (X - Y))
# surf = ax.plot_surface(X,
#                        Y,
#                        Z,
#                        cmap=cm.coolwarm,
#                        linewidth=0,
#                        antialiased=False)
wire = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
# surf.set_facecolor((0,0,0,0))
ax.set_xlabel("$u_j$")
ax.set_ylabel("$u_i$")
ax.set_zlabel("$L'_{RANK}(i,j)$")
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()
ax.xaxis.set_major_locator(LinearLocator(4))
ax.yaxis.set_major_locator(LinearLocator(4))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.01f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.01f'))
plt.savefig("../Masters_Thesis/New_Thesis/masters-thesis/gfx/hinge_original.pdf")
plt.clf()
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# surf = ax.plot_surface(X,
#                        Y,
#                        Z_squared,
#                     #    cmap=cm.coolwarm,
#                        linewidth=0,
#                        antialiased=False)
fig = plt.figure(figsize=plt.figaspect(0.8))
ax = fig.add_subplot(111, projection='3d')
Z_squared = np.maximum(0, epsilon - (X - Y))**2
wire = ax.plot_wireframe(X, Y, Z_squared, rstride=2, cstride=2)
# surf.set_facecolor((0,0,0,0))
ax.set_xlabel("$u_j$")
ax.set_ylabel("$u_i$")
ax.set_zlabel("$L_{RANK}(i,j)$")

# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(5))
ax.xaxis.set_major_locator(LinearLocator(4))
ax.yaxis.set_major_locator(LinearLocator(4))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.01f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.01f'))

# Add a color bar which maps values to colors.

plt.savefig("../Masters_Thesis/New_Thesis/masters-thesis/gfx/hinge_squared.pdf")
