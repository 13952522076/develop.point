"""
for training new model with pointnet2_ops
Usage:
python main_new.py --use_normals --use_uniform_sample --model new1Amax
or
nohup python classify.py --model new1A > new_nohup/PCTNEW.out &
"""
import argparse
import os
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args
from data import ModelNet40
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
from helper import cal_loss
import numpy as np

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name]))



"""Parameters"""
parser = argparse.ArgumentParser('training')
# parser.add_argument('-d', '--data_path', default='data/modelnet40_normal_resampled/', type=str)
parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH', default="./checkpoints/",
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--msg', type=str, help='message after checkpoint')
parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
parser.add_argument('--model', default='model21H', help='model name [default: pointnet_cls]')
parser.add_argument('--num_classes', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
parser.add_argument('--epoch', default=350, type=int, help='number of epoch in training')
parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
# parser.add_argument('--use_normals', action='store_true', default=False, help='use normals besides x,y,z')
# parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
# parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')


args = parser.parse_args()


foldsList = os.listdir(args.checkpoint)
assert len(foldsList) > 0, f"no folders found in current path: {args.checkpoint}"
modellist = []
for folder in foldsList:
    if folder.startswith(args.model):
        modellist.append(folder)
assert len(modellist) > 0, f"no model folders found"
print(f"Found {len(modellist)} pretrained model folders.")

best_acc_mean = 0.0
best_acc_all = 0.0
best_acc_mean_model_path = ""
best_acc_all_model_path = ""
for model_path in modellist:
    model_path = os.path.join(args.checkpoint, model_path)
    log_path = os.path.join(model_path, "log.txt")
    logs = np.loadtxt(log_path, skiprows=1)
    acc_mean = max(logs[:,-2])
    acc_all = max(logs[:, -1])
    if acc_mean>best_acc_mean:
        best_acc_mean = acc_mean
        best_acc_mean_model_path = model_path

    if acc_all>best_acc_all:
        best_acc_all = acc_all
        best_acc_all_model_path = model_path

print(f"Best mean accuracy is: {best_acc_mean} | [{best_acc_mean_model_path}]")
print(f"Best all  accuracy is: {best_acc_all} | [{best_acc_all_model_path}]")


