import numpy as np
from scipy.stats import gumbel_r
import matplotlib.pyplot as plt
import seaborn as sbs

# fig, ax = plt.subplots(1, 1)

mu1 = 8
mu2 = 3
mu3 = -8

figures_path = "../Masters_Thesis/New_Thesis/masters-thesis/gfx/"

fig,axes=plt.subplots(1,1)

x = np.linspace(-13, 20, 10000)
print(x)
y1 = gumbel_r.pdf(x, mu1, 2)
y2 = gumbel_r.pdf(x, mu2, 2)
y3 = gumbel_r.pdf(x, mu3, 2)
axes.set_xticks([mu1,mu2,mu3],["$\\mu_1$","$\\mu_2$","$\\mu_3$"])
axes.tick_params(labelsize=16)
plt.xticks([mu1,mu2,mu3],["$\\mu_1$","$\\mu_2$","$\\mu_3$"])
# axes.margins(x=-0.25,y=-0)
# plt.yticks([],[])
axes.plot(x,y1, color="C2")
axes.plot(x,y2, color="C0")
axes.plot(x,y3, color="C3")
axes.fill_between(x,y1, alpha=0.3, color="C2")
axes.fill_between(x,y2, alpha=0.3, color="C0")
axes.fill_between(x,y3, alpha=0.3, color="C3")
# plt.axes().set_aspect(100)
axes.set_yticks([],[])
fig.tig
fig.set_size_inches(10.5, 3.5)
plt.savefig(figures_path+"gumbel_thurstone.pdf")