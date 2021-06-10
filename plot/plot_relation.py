"""
Plot the relationship of local-transformer and global-transformer.
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
    # parser.add_argument('-d', '--data_path', default='data/modelnet40_normal_resampled/', type=str)
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        default="/Users/melody/Downloads/checkpoint/model21H-seed6test6/best_checkpoint.pth",
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--model', default='plot21H', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_points', type=int, default=2048, help='Point Number')


    # for ploting  2108chair    1938airplane   11 van
    parser.add_argument('--id', default=1938, type=int, help='ID of the example 2468')
    parser.add_argument('--stage', type=int, default=0, help='index of stage')
    parser.add_argument('--point_id', type=int, default=10, help='index of selected point in FPS')
    parser.add_argument('--head_id', type=int, default=2, help='index of selected head')
    parser.add_argument('--save', action='store_true', default=False, help='use normals besides x,y,z')
    parser.add_argument('--show', action='store_true', default=False, help='use normals besides x,y,z')
    return parser.parse_args()

def plot_xyz(xyz, args, selected_xyz=None, name="figure.pdf", local_attention=None, global_attention=None, plot_which=None ): # xyz: [n,3] selected_xyz:[3]
    fig = pyplot.figure()
    ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    x_vals = xyz[:, 0]
    y_vals = xyz[:, 1]
    z_vals = xyz[:, 2]

    ax.set_xlim3d(min(x_vals)*0.9, max(x_vals)*0.9)
    ax.set_ylim3d(min(y_vals)*0.9, max(y_vals)*0.9)
    ax.set_zlim3d(min(z_vals)*0.9, max(z_vals)*0.9)


    if plot_which=="local":
        assert local_attention is not None
        print(f"Local  attention range: {local_attention.min()}-{local_attention.max()}")
        norm = pyplot.Normalize(vmin=-0.6, vmax=local_attention.max()*1.1)
        ax.scatter(x_vals, y_vals, z_vals, c=local_attention, cmap='Reds', norm=norm)
    elif plot_which=="global":
        assert global_attention is not None
        print(f"Global attention range: {global_attention.min()}-{global_attention.max()}")
        print(f"Global attention max value and indices: {global_attention.max(dim=0)}")
        norm = pyplot.Normalize(vmin=-0.6, vmax=global_attention.max()*1.1)
        # print(f"global norm_color range: {norm_color.min()}-{norm_color.max()}")
        ax.scatter(x_vals, y_vals, z_vals, c=global_attention, cmap='Reds', norm=norm)
    else:
        ax.scatter(x_vals, y_vals, z_vals, color="mediumseagreen")
    if selected_xyz is not None:
        ax.scatter(selected_xyz[0],selected_xyz[1], selected_xyz[2], color="green", marker="*", s=150,)
    # # make the panes transparent
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # # make the grid lines transparent
    # ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)


    ax.set_axis_off()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # pyplot.tight_layout()
    if args.show:
        pyplot.show()
    if args.save:
        fig.savefig(name, bbox_inches='tight', pad_inches=0.00, transparent=True)

    pyplot.close()

def plot_struct(struct, args):
    # print(f'struct["previous_xyz"]: {(struct["previous_xyz"]).shape}')  # [1, 256, 3]
    # print(f'struct["xyz"]:          {(struct["xyz"]).shape}')  # [1, 64, 3]
    # print(f'struct["grouped_xyz"]: {(struct["grouped_xyz"]).shape}')  # [1, 64, 32, 3]
    # print(f'struct["local_attention"]: {(struct["local_attention"]).shape}')  # [64, 4, 32, 32]
    # print(f'struct["global_attention"]: {(struct["global_attention"]).shape}')  # [1, 4, 64, 64]
    id = args.point_id
    head_id = args.head_id
    xyz = struct["xyz"][0]  # [p,3]
    selected_xyz = xyz[id]  # [3]
    neigbor_xyz = struct["grouped_xyz"][0,id,:,:]  # [k,3]
    local_attention = struct["local_attention"][id, head_id, 0, :] # [k]
    global_attention = struct["global_attention"][0, head_id, id, :] # [p]

    #f"attention/Image-{args.id}_Point-{args.point_id}_Stage-{args.stage}_Local/Global_Head-{args.head_id}.pdf"
    #plot_xyz(xyz, args, selected_xyz=None, name="figure.pdf", local_attention=None, global_attention=None, plot_which=None )
    plot_xyz(neigbor_xyz.data, args, selected_xyz,
             name= f"attention/Image-{args.id}_Point-{args.point_id}_Stage-{args.stage}_Local_Head-{args.head_id}.pdf",
             local_attention=local_attention.data,  global_attention = global_attention.data, plot_which="local")
    plot_xyz(xyz.data, args, selected_xyz,
             name=f"attention/Image-{args.id}_Point-{args.point_id}_Stage-{args.stage}_Global_Head-{args.head_id}.pdf",
             local_attention=local_attention.data, global_attention=global_attention.data, plot_which="global")

def main():
    args = parse_args()
    # print(f"args: {args}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    print("===> building and loading models ...")
    net = plot21H()
    # print(net)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    new_check_point = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:] # remove `module.`
        new_check_point[name] = v

    net.load_state_dict(new_check_point)
    net.eval()

    print('==> Preparing data ...')
    # train_set =ModelNet40(partition='train', num_points=args.num_points)
    test_set = ModelNet40(partition='test', num_points=args.num_points)

    data, label = test_set.__getitem__(args.id)
    plot_xyz(data, args, None, name=f"attention/Image-{args.id}.pdf" )
    data = torch.tensor(data).unsqueeze(dim=0)
    data = data.permute(0, 2, 1)
    with torch.no_grad():
        logits, structure_list = net(data)
    preds = logits.max(dim=1)[1]
    print(f"predict: {preds} | label: {label}")

    struct = structure_list[args.stage]
    plot_struct(struct, args)



if __name__ == '__main__':
    set_seed(32) # must
    main()






"""
id = np.random.randint(0,2048)
id=800 #airplane 11
id=2001 # lighter
id=860
color='lightskyblue'
color='yellowgreen'
color='orange'
points=50
save_fig=True
rotation=True
scale=True


datset = ModelNet40(points, partition='test')
sample,label = datset.__getitem__(id)




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


ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, color = color)


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
"""
