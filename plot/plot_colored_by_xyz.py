"""
Plot the relationship (mean of heads) for global Transformer.
Plot the relationship of local-transformer and global-transformer.
python3 plot_relation2.py --id 26 --point_id 10 --stage 0 --save
"""
import argparse
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import numpy as np
from collections import OrderedDict
import h5py
import math
import sys
sys.path.append("..")
from data import ModelNet40
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
# import models as models
from plot21 import plot21H
from utils import set_seed









def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--num_points', type=int, default=5000, help='Point Number')

    # for ploting 26 airplane
    parser.add_argument('--id', default=800, type=int, help='ID of the example 2468')
    parser.add_argument('--save', action='store_true', default=False, help='use normals besides x,y,z')
    parser.add_argument('--show', action='store_true', default=True, help='use normals besides x,y,z')
    return parser.parse_args()

def plot_xyz(xyz, args,  name="figure.pdf" ): # xyz: [n,3] selected_xyz:[3]
    fig = pyplot.figure()
    ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    x_vals = xyz[:, 0]
    y_vals = xyz[:, 1]
    z_vals = xyz[:, 2]

    ax.set_xlim3d(min(x_vals)*0.9, max(x_vals)*0.9)
    ax.set_ylim3d(min(y_vals)*0.9, max(y_vals)*0.9)
    ax.set_zlim3d(min(z_vals)*0.9, max(z_vals)*0.9)

    color = x_vals+y_vals+z_vals
    norm = pyplot.Normalize(vmin=min(color), vmax=max(color))
    ax.scatter(x_vals, y_vals, z_vals, c=color, cmap='hsv', norm=norm)

    ax.set_axis_off()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # pyplot.tight_layout()
    if args.show:
        pyplot.show()
    if args.save:
        fig.savefig(name, bbox_inches='tight', pad_inches=0.00, transparent=True)

    pyplot.close()

def main():
    args = parse_args()
    # print(f"args: {args}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    print('==> Preparing data ...')
    # train_set =ModelNet40(partition='train', num_points=args.num_points)
    test_set = ModelNet40(partition='test', num_points=args.num_points)

    data, label = test_set.__getitem__(args.id)
    plot_xyz(data, args, name=f"forstructure/Image-{args.id}-{args.num_points}.pdf" )



if __name__ == '__main__':
    set_seed(32) # must
    main()

