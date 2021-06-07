import matplotlib.pyplot as plt
import numpy as np

n = 1024  # data size
X = np.random.normal(0, 1, n)  # 每一个点的X值,平均数是0，方差是1
Y = np.random.normal(0, 1, n)  # 每一个点的Y值

# 这里我们每个点的颜色和该点的X值+Y值的和相关
color = X + Y

# 使用我们上面说的灰度图
cmap = plt.get_cmap('Greys')
# cmap = plt.cm.Greys #也可以这么写
# 利用normlize来标准化颜色的值
norm = plt.Normalize(vmin=-3, vmax=3)

# 散点图
plt.scatter(X, Y, s=75, alpha=0.5, c=color, cmap=cmap, norm=norm)

plt.xlim(-1.5, 1.5)
plt.xticks(())  # ignore xticks
plt.ylim(-1.5, 1.5)
plt.yticks(())  # ignore yticks

plt.show()




norm = plt.Normalize(vmin=-30, vmax=3)

# 散点图
plt.scatter(X, Y, s=75, alpha=0.5, c=color, cmap=cmap, norm=norm)

plt.xlim(-1.5, 1.5)
plt.xticks(())  # ignore xticks
plt.ylim(-1.5, 1.5)
plt.yticks(())  # ignore yticks

plt.show()
