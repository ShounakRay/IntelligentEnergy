# @Author: Shounak Ray <Ray>
# @Date:   17-Mar-2021 16:03:89:898  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: heatmap_anitest.py
# @Last modified by:   Ray
# @Last modified time: 17-Mar-2021 23:03:04:045  GMT-0600
# @License: [Private IP]

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig = plt.figure()
data = np.random.rand(10, 10)
sns.heatmap(data, square=True)


def init():
    sns.heatmap(np.zeros((10, 10)), square=True, cbar=False)


def animate(i):
    data = np.random.rand(10, 10)
    sns.heatmap(data, square=True, cbar=False)


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=60, repeat=False)
try:
    writer = animation.writers['ffmpeg']
except KeyError:
    writer = animation.writers['avconv']
writer = writer(fps=60)
anim.save('CrossCorrelation/test_empty.mp4', writer=writer, dpi=100)
