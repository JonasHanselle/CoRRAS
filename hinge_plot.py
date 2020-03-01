from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

fig = plt.figure(figsize=plt.figaspect(0.8))
ax = fig.add_subplot(111, projection='3d')

y_hat_i = np.arange(-2, 2, 0.09)
y_hat_j = np.arange(-2, 2, 0.09)
X, Y = np.meshgrid(y_hat_i, y_hat_j)
epsilon = 1
Z = np.maximum(0, epsilon - (X - Y))
Z_squared = np.maximum(0, epsilon - (X - Y))**2
# surf = ax.plot_surface(X,
#                        Y,
#                        Z,
#                        cmap=cm.coolwarm,
#                        linewidth=0,
#                        antialiased=False)
wire = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, cmap=cm.coolwarm)
# surf.set_facecolor((0,0,0,0))
ax.set_xlabel("$\hat{y}_j$")
ax.set_ylabel("$\hat{y}_i$")
ax.set_zlabel("$L(i,j)$")
# fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
plt.savefig()
# ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(X,
                       Y,
                       Z_squared,
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False)
wire = ax.plot_wireframe(X, Y, Z_squared, rstride=2, cstride=2)
# surf.set_facecolor((0,0,0,0))
ax.set_xlabel("$\hat{y}_j$")
ax.set_ylabel("$\hat{y}_i$")
ax.set_zlabel("$L(i,j)$")

# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.

plt.savefig("margin_plot.pdf")
