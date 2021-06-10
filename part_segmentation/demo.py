import numpy as np
import matplotlib.pyplot as plt

cmap = plt.get_cmap('viridis')
names = ["bob", "joe", "andrew", "pete"]
colors = cmap(np.linspace(0, 1, len(names)))
print(colors)
print(colors.shape)
# [[ 0.267004  0.004874  0.329415  1.      ]
#  [ 0.190631  0.407061  0.556089  1.      ]
#  [ 0.20803   0.718701  0.472873  1.      ]
#  [ 0.993248  0.906157  0.143936  1.      ]]

x = np.linspace(0, np.pi*2, 100)
for i, (name, color) in enumerate(zip(names, colors), 1):
    print(color)
    print(color.shape)
    plt.plot(x, np.sin(x)/i, label=name, c=color)
plt.legend()
plt.show()
