from util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
set_seed(1996)

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
from dataset import get_loader
from criterion import Eval
import torchvision.utils as vutils
from apex import amp
from models.MineNet import MineNet
from loss import BlcBCE,BCE
import torch.nn.functional as F

# Parameter from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model',
                    default='CoSalNet',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--loss',
                    default='DSLoss_IoU_noCAM',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--bs', '--batch_size', default=1, type=int)
parser.add_argument('--lr',
                    '--learning_rate',
                    default=0.0001,
                    type=float,
                    help='Initial learning rate')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='Jigsaw2_DUTS',
                    type=str,
                    help="Options: 'Jigsaw2_DUTS', 'DUTS_class'")
parser.add_argument('--valset',
                    default='CoSal15',
                    type=str,
                    help="Options: 'CoSal15', 'CoCA'")
parser.add_argument('--size', default=224, type=int, help='input size')
parser.add_argument('--tmp', default=None, help='Temporary folder')
parser.add_argument("--use_tensorboard", action='store_true')
parser.add_argument("--jigsaw", action='store_true')

args = parser.parse_args()

# Init TensorboardX
if args.use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tmp)

if args.trainset == 'Jigsaw2_DUTS':
    # train_img_path = './data/train/classfuse84final/img/'
    # train_gt_path = './data/train/classfuse84final/gt/'
    # train_edge_path = './data/train/classfuse84final/edge/'
    train_img_path = './data/train/classfuse84final/img/'
    train_gt_path = './data/train/classfuse84final/gt/'
    train_edge_path = './data/train/classfuse84final/edge/'
    train_loader = get_loader(train_img_path,
                              train_gt_path,
                              train_edge_path,
                              args.size,
                              args.bs,
                              max_num=20,
                              istrain=True,
                              jigsaw=args.jigsaw,
                              shuffle=False,
                              num_workers=4,
                              pin=True)
else:
    print('Unkonwn train dataset')
    print(args.dataset)

if args.valset == 'CoSal15':
    val_img_path1 = './data/test/CoSal2015test1/img/'
    val_gt_path1 = './data/test/CoSal2015test1/gt/'
    val_loader1 = get_loader(val_img_path1,
                            val_gt_path1,
                            val_gt_path1,
                            args.size,
                            1,
                            istrain=False,
                            jigsaw=args.jigsaw,
                            shuffle=False,
                            num_workers=4,
                            pin=True)
    val_img_path2 = './data/test/CoCAtest1/img/'
    val_gt_path2 = './data/test/CoCAtest1/gt/'
    val_loader2 = get_loader(val_img_path2,
                            val_gt_path2,
                            val_gt_path2,
                            args.size,
                            1,
                            istrain=False,
                            jigsaw=args.jigsaw,
                            shuffle=False,
                            num_workers=4,
                            pin=True)
else:
    print('Unkonwn val dataset')
    print(args.dataset)

# make dir for tmp
os.makedirs(args.tmp, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.tmp, "log.txt"))

# Init model
device = torch.device("cuda")

model = eval('MineNet()')
model = model.to(device)

## parameter
head_params = model.coatten.parameters()
backbone_params = list(map(id, model.iife.backbone.parameters()))
base_params = filter(lambda p: id(p) not in backbone_params,
                     model.iife.parameters())

# Setting optimizer
optimizer = optim.Adam(params=[{'params':head_params},{'params':base_params}], lr=args.lr, betas=[0.9, 0.99])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Setting Loss
exec('from loss import ' + args.loss)
dsloss = eval(args.loss + '()')
bbceloss = eval('BlcBCE()')
bceloss = eval('BCE()')

def main():
    val_mae_record1, val_mae_record2 = [], []
    val_Sm_record1, val_Sm_record2 = [], []

    print(args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(epoch)
        [val_mae1, val_Sm1, val_mae2, val_Sm2] = validate(epoch)

        val_mae_record1.append(val_mae1)
        val_Sm_record1.append(val_Sm1)
        val_mae_record2.append(val_mae2)
        val_Sm_record2.append(val_Sm2)
        
        # Save checkpoint
        tanxnet_dict = dict(**model.coatten.state_dict(), **(model.iife.state_dict()))
        torch.save(tanxnet_dict, os.path.join(args.tmp, "checkpoint.pth"))

        torch.save(tanxnet_dict, os.path.join(args.tmp, "checkpoint_%d.pth"%epoch))

    # Show in tensorboard
    if args.use_tensorboard:
        writer.add_scalar('Loss/total', train_loss, epoch)

        writer.add_scalar('Metric/MAE', val_mae, epoch)
        writer.add_scalar('Metric/Sm', val_Sm, epoch)


def train(epoch):
    loss_log = AverageMeter()

    # Switch to train mode
    model.train()
    model.set_mode('train')

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device)
        gts = batch[1].to(device)
        edges = batch[2].to(device)

        scaled_preds, scaled_edges = model(inputs)
        loss1 = dsloss(scaled_preds, gts)
        loss2 = bbceloss(scaled_edges, edges)

        loss = loss1 + loss2

        loss_log.update(loss, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]  '
                        'Train Loss: {loss.val:.3f} ({loss.avg:.3f})  '.format(
                            epoch,
                            args.epochs,
                            batch_idx,
                            len(train_loader),
                            loss=loss_log,
                        ))
            print("lr：%.8f" % (optimizer.param_groups[0]['lr']))
    scheduler.step()
    logger.info('@==Final== Epoch[{0}/{1}]  '
                'Train Loss: {loss.avg:.3f}  '.format(epoch,
                                                      args.epochs,
                                                      loss=loss_log))

    return loss_log.avg


def validate(epoch):
    # Switch to evaluate mode
    model.eval()
    model.set_mode('test')

    saved_root1 = os.path.join(args.tmp, 'Salmaps1')
    # make dir for saving results
    os.makedirs(saved_root1, exist_ok=True)

    for batch in tqdm(val_loader1):
        inputs = batch[0].to(device)
        subpaths = batch[1]
        ori_sizes = batch[2]

        scaled_preds, edges = model(inputs)

        num = len(scaled_preds)

        os.makedirs(os.path.join(saved_root1, subpaths[0][0].split('/')[0]),
                    exist_ok=True)

        for inum in range(num):
            subpath = subpaths[inum][0]
            ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
            res = nn.functional.interpolate(scaled_preds[inum][-1],
                                            size=ori_size,
                                            mode='bilinear',
                                            align_corners=True)
            save_tensor_img(res, os.path.join(saved_root1, subpath))

    evaler = Eval(pred_root=saved_root1, label_root=val_gt_path1)
    mae1 = evaler.eval_mae()
    Sm1 = evaler.eval_Smeasure()

    saved_root2 = os.path.join(args.tmp, 'Salmaps2')
    # make dir for saving results
    os.makedirs(saved_root2, exist_ok=True)

    for batch in tqdm(val_loader2):
        inputs = batch[0].to(device)
        subpaths = batch[1]
        ori_sizes = batch[2]

        scaled_preds, edges = model(inputs)

        num = len(scaled_preds)

        os.makedirs(os.path.join(saved_root2, subpaths[0][0].split('/')[0]),
                    exist_ok=True)

        for inum in range(num):
            subpath = subpaths[inum][0]
            ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
            res = nn.functional.interpolate(scaled_preds[inum][-1],
                                            size=ori_size,
                                            mode='bilinear',
                                            align_corners=True)
            save_tensor_img(res, os.path.join(saved_root2, subpath))

    evaler = Eval(pred_root=saved_root2, label_root=val_gt_path2)
    mae2 = evaler.eval_mae()
    Sm2 = evaler.eval_Smeasure()

    logger.info('@==Final== Epoch[{0}/{1}]  '
                'Cosal2015: MAE: {mae1:.3f}  '
                'Sm: {Sm1:.3f}  '
                'CoCA: MAE: {mae2:.3f}  '
                'Sm: {Sm2:.3f}'.format(epoch, args.epochs, mae1=mae1, Sm1=Sm1, mae2=mae2,Sm2=Sm2))

    return mae1, Sm1, mae2, Sm2


if __name__ == '__main__':
    main()
