import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设函数为y=3x-2
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-2.0, 2.1, 0.1):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
        mse_list.append((l_sum / 3))
    w_list.append(w)

fig = plt.figure()
ax = fig.add_axes(Axes3D(fig))

x = np.array(w_list)
y = np.arange(-2.0, 2.1, 0.1)
z = np.reshape(mse_list, (x.__len__(), y.__len__()))
x, y = np.meshgrid(x, y)
ax.plot_surface(x, y, z)
plt.show()
