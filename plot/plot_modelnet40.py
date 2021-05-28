from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
import h5py
import math


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the user specified axis by theta radians.
    """
    # convert the input to an array
    axis = np.asarray(axis)
    # Get unit vector of our axis
    axis = axis/math.sqrt(np.dot(axis, axis))
    # take the cosine of out rotation degree in radians
    a = math.cos(theta/2.0)
    # get the rest rotation matrix components
    b, c, d = -axis*math.sin(theta/2.0)
    # create squared terms
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    # create cross terms
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    # return our rotation matrix

    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

f = h5py.File("/Users/melody/Downloads/ply_data_test0.h5", 'r')
data = f["data"]
id = np.random.randint(0,2048)
id=800 #airplane 11
id=2001 # lighter
id=1
points=2048
save_fig=True
sample = data[id,0:points,:]

# rotation = np.array([[1., 0.,0.], [0.,1.,0.], [0.,0.,1.]])
rotation = rotation_matrix([0.3, 0., 0.], 1.2)
sample= np.matmul(sample, rotation)

fig = pyplot.figure()
ax = Axes3D(fig)

sequence_containing_x_vals = sample[:, 0]
x_min = min(sequence_containing_x_vals)
x_max = max(sequence_containing_x_vals)
print(f"x range: {x_max-x_min}")
sequence_containing_y_vals = sample[:, 1]
y_min = min(sequence_containing_y_vals)
y_max = max(sequence_containing_y_vals)
print(f"y range: {y_max-y_min}")
sequence_containing_z_vals = sample[:, 2]
z_min = min(sequence_containing_z_vals)
z_max = max(sequence_containing_z_vals)
print(f"z range: {z_max-z_min}")


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
# pyplot.tight_layout()
pyplot.show()
if save_fig:
    fig.savefig(f"{id}_{points}.pdf", bbox_inches='tight', pad_inches=0.05, transparent=True)
