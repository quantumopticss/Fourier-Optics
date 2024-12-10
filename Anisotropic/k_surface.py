import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import root_scalar

# 定义折射率（各向异性）
n1 = 1.5  # x方向的折射率 n11
n2 = 1.2  # y方向的折射率 n22
n3 = 1.8  # z方向的折射率 n33

# 自由空间波数
k0 = 1.0  # 可以设为1，或者根据需要设定

# 定义波矢空间
k1_list = np.linspace(0, n1, 20)
k2_list = np.linspace(0, n2, 20)
k1, k2 = np.meshgrid(k1_list, k1_list)

k3 = np.empty_like(k1)

# 存储满足行列式为零的k3值
k3_positive = []

# 将行列式设为函数，关于k3
def d_k3(k3,k1,k2):
    f = ( (n1*n2*n3)**2 - (n1*n3)**2*(k1**2 + k3**2) - (n2*n3)**2*(k2**2 + k3**2) - (n1*n2)**2*(k1**2 + k2**2) 
            + (k1*k2)**2*(n1**2 + n2**2) + (k1*k3)**2*(n1**2 + n3**2) + (k2*k3)**2*(n2**2 + n3**2) + n3**2*k3**4 + n2**2*k2**4 + n1**2*k1**4
    )
    return f

for i in range(len(k1_list)):
    for j in range(len(k2_list)):
        k1_val = k1_list[i]
        k2_val = k2_list[j]
    
        res = root_scalar(d_k3,args = (k1_val,k2_val),method = "newton", bracket = (0.,n3), x0 = n3/2)
        k3[i,j] = res.root

# 转换为数组


# 绘制 k 表面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(k1, k2, k3, cmap='viridis', edgecolor='none', alpha=0.8)

# 设置标签和标题
ax.set_xlabel('k1')
ax.set_ylabel('k2')
ax.set_title('k-Surface of Anisotropic Medium')
plt.show()
