from torch import nn
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable

class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            #compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(
                pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1

            #IoU loss is (1-IoU1)
            IoU = IoU + (1 - IoU1)

        return IoU / b


class DSLoss_IoU_noCAM(nn.Module):
    def __init__(self):
        super(DSLoss_IoU_noCAM, self).__init__()
        self.iou = IoU_loss()

    def forward(self, scaled_preds, gt):
        [_, num, _, _, _] = gt.size()
        loss = 0
        for inum in range(num):
            for isout, x in enumerate(scaled_preds[inum][1:]):
                loss += self.iou(x, gt[:, inum, :, :, :])
        return loss

class bce2d_new(torch.nn.Module):
    def __init__(self):
        super(bce2d_new, self).__init__()

    def forward(self, input, target, reduction=None):
        # assert(input.size() == target.size())
        pos = torch.eq(target, 1).float()
        neg = torch.eq(target, 0).float()

        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        num_total = num_pos + num_neg

        alpha = num_neg  / num_total
        beta = 1.1 * num_pos  / num_total
        weights = alpha * pos + beta * neg

        return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

class BlcBCE(torch.nn.Module):
    def __init__(self):
        super(BlcBCE, self).__init__()

        self.bce = bce2d_new()
    
    def forward(self, scaled_preds, gt):
        [_, num, _, _, _] = gt.size()
        loss = 0
        for inum in range(num):
            for isout, x in enumerate(scaled_preds[inum][1:]):
                loss += self.bce(x, gt[:, inum, :, :, :], reduction='elementwise_mean')
        return loss

class BCE(torch.nn.Module):
    def __init__(self):
        super(BCE, self).__init__()

        self.bce = bce2d_new()
    
    def forward(self, scaled_preds, gt):
        [_, num, _, _, _] = gt.size()
        loss = 0
        for inum in range(num):
            for isout, x in enumerate(scaled_preds[inum][1:]):
                loss += F.binary_cross_entropy_with_logits(x, gt[:, inum, :, :, :], reduction='elementwise_mean')
        return loss