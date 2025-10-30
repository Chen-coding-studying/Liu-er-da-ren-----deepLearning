import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w + b   # w weight

def loss(x, y):
    y_pred = forward(x) 
    return (y_pred - y) ** 2   

w_list = [] # save weights
b_list = [] # save biases
mse_list = [] # save cost values

for w in np.arange(0.0, 4.1, 0.01):
    for b in np.arange(-1.0, 1.2, 0.02):
        print(f"Weight: {w}, Bias: {b}")
        l_sum = 0

        for x_val, y_val in zip(x_data, y_data):# zip 同时访问 x_data 和 y_data

            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            
            print('\t', x_val, y_val, y_pred_val, loss_val)
    
        print('MSE=', l_sum / 3)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum / 3) 
    


# 转成 numpy 数组
w_list = np.array(w_list)
b_list = np.array(b_list)
mse_list = np.array(mse_list)

# 为绘制3D曲面做形状调整
W, B = np.meshgrid(
    np.arange(0.0, 4.1, 0.01),
    np.arange(-1.0, 1.2, 0.02)
)

MSE = mse_list.reshape(B.shape)  # 保证对应形状一致

# 绘制 3D 曲面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, MSE, cmap='viridis', alpha=0.8)

ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('MSE Loss')

plt.show()

