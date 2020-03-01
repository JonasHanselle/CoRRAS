import numpy as np
from scipy.stats import gumbel_r
import matplotlib.pyplot as plt
import seaborn as sbs

# fig, ax = plt.subplots(1, 1)

mu1 = 13
mu2 = 0
mu3 = -5

x = np.linspace(-13, 35, 10000)
print(x)
y1 = gumbel_r.pdf(x, mu1, 2)
y2 = gumbel_r.pdf(x, mu2, 2)
y3 = gumbel_r.pdf(x, mu3, 2)
plt.xticks([mu1,mu2,mu3],["$\\mu_1$","$\\mu_2$","$\\mu_3$"])
# plt.yticks([],[])
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.fill_between(x,y1, alpha=0.3)
plt.fill_between(x,y2, alpha=0.3)
plt.fill_between(x,y3, alpha=0.3)
plt.axes().set_aspect(100)
plt.savefig("gumbel_thurstone.pdf")