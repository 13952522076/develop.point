from plyfile import PlyData, PlyElement
import random
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D



plydata = PlyData.read('/Users/melody/Downloads/bunny/data/bun000.ply')

points = 4000

print(plydata.elements[0].properties)
max_num = plydata.elements[0].count
print(max_num)
# sets = plydata['vertex']
sets = plydata.elements[0]

fig = pyplot.figure()
ax = Axes3D(fig)

for i in range(0,points):
    idx = random.randint(0,max_num)
    data = sets[idx]
    # print(f"{data[0]} {data[1]} {data[2]}")
    ax.scatter(data[0], data[1], data[2], c="green")
ax.set_axis_off()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
pyplot.show()
pyplot.close()
