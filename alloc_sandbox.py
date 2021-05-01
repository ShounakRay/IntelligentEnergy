# @Author: Shounak Ray <Ray>
# @Date:   30-Apr-2021 12:04:07:077  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: alloc_sandbox.py
# @Last modified by:   Ray
# @Last modified time: 01-May-2021 11:05:80:802  GMT-0600
# @License: [Private IP]


import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')

if __name__ == '__main__':
    import matplotlib.pyplot as plt


M = 200
m = 100
b = 10


b_range = 1000


def func(x, M=M, b=b, m=m):
    if not M > m:
        raise Exception('BAD')
    return (M + b - x) / (M + b - m)


points = {}
b_iter = np.linspace(1, b_range, num=100)
for b in b_iter:
    iter = np.linspace(m, M, num=100)
    points[b] = dict(zip(iter, [func(x, b=b) for x in iter]))

fig, ax = plt.subplots(figsize=(12, 10))
ax = plt.axes(projection='3d')
for b in b_iter:
    x = list(points.get(b).keys())
    y = list(points.get(b).values())
    z = b
    ax.scatter3D(x, y, z, c=x)
ax.set_xlabel('data points (x)', fontsize=10)
ax.set_ylabel('steam proportion (func)', fontsize=10)
ax.set_zlabel('base (b)', fontsize=10)
plt.tight_layout()
plt.show()
