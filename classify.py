"""
for training new model with pointnet2_ops
Usage:
python main_new.py --use_normals --use_uniform_sample --model new1Amax
or
nohup python classify.py --model PCT > new_nohup/PCTNEW.out &
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


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    # parser.add_argument('-d', '--data_path', default='data/modelnet40_normal_resampled/', type=str)
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='PCT', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=250, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    # parser.add_argument('--use_normals', action='store_true', default=False, help='use normals besides x,y,z')
    # parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    # parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(args.seed)
    else:
        device = 'cpu'
    print(f"==> Using device: {device}")
    if args.checkpoint is None:
        time_stamp = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
        args.checkpoint = 'checkpoints/' + args.model + time_stamp
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate', 'Train-Loss', 'Train-acc', 'Valid-Loss', 'Valid-acc'])

    print('==> Preparing data..')
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)

    # Model
    print('==> Building model..')
    net = models.__dict__[args.model]()
    criterion = cal_loss
    net = net.to(device)
    # criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate / 100)

    best_test_acc = 0.  # best test accuracy
    best_train_acc = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    for epoch in range(start_epoch, args.epoch):
        print('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(net, train_loader, optimizer, criterion, device)  # {"loss", "acc", "time"}
        # test_out = validate(net, test_loader, criterion, device)
        scheduler.step()

        # if test_out["acc"] > best_test_acc:
        #     best_test_acc = test_out["acc"]
        #     is_best = True
        # else:
        #     is_best = False
        #
        # best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        # best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        # best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        # best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss
        #
        # save_model(net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best)
        # logger.append([epoch, optimizer.param_groups[0]['lr'],
        #                train_out["loss"], train_out["acc"],
        #                test_out["loss"], test_out["acc"]])
        # print(f"Training loss:{train_out['loss']} acc:{train_out['acc']}% time:{train_out['time']}s) | "
        #       f"Testing loss:{test_out['loss']} acc:{test_out['acc']}% time:{test_out['time']}s) \n\n")
    logger.close()


    print(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    print(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    print(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    print(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    print(f"++++++++" * 5)


def train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        optimizer.zero_grad()
        logits = net(data)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())


        total += label.size(0)
        correct += preds.eq(label).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (points, targets) in enumerate(testloader):
            points = points.data.numpy()
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            points, targets = points.to(device), targets.to(device).long()
            out = net(points)
            loss = criterion(out, targets)
            test_loss += loss.item()
            _, predicted = out["logits"].max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * correct / total)),
        "time": time_cost
    }


if __name__ == '__main__':
    main()
