import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 8.0, 11.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


W = np.arange(0.0, 4.1, 0.1)
B = np.arange(0.0, 4.1, 0.1)
[w, b] = np.meshgrid(W, B)
# print(w)
# print(b)

# l_sum记录了损失矩阵
l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)
    print(y_pred_val)
    loss_val = loss(x_val, y_val)
    l_sum += loss_val

print(l_sum)
fig = plt.figure()
ax = fig.add_axes(Axes3D(fig))
ax.plot_surface(w, b, l_sum / 3)
plt.show()
