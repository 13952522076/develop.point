from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
import h5py

f = h5py.File("/Users/melody/Downloads/ply_data_test0.h5", 'r')
data = f["data"]
id = np.random.randint(0,2048)
id=800 #airplane 11
id=2001 # lighter
points=2000
sample = data[id,0:points,:]


fig = pyplot.figure()
ax = Axes3D(fig)

sequence_containing_x_vals = sample[:, 0]
x_min = min(sequence_containing_x_vals)
x_max = max(sequence_containing_x_vals)
sequence_containing_y_vals = sample[:, 1]
y_min = min(sequence_containing_y_vals)
y_max = max(sequence_containing_y_vals)
sequence_containing_z_vals = sample[:, 2]
z_min = min(sequence_containing_z_vals)
z_max = max(sequence_containing_z_vals)


ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)


# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
# Make panes transparent
ax.set_xlim3d(x_min,x_max)
ax.set_ylim3d(y_min,y_max)
ax.set_zlim3d(z_min,z_max)

ax.set_axis_off()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
pyplot.tight_layout()
pyplot.show()
fig.savefig(f"{id}_{points}.pdf", bbox_inches='tight', pad_inches=0.05, transparent=True)
