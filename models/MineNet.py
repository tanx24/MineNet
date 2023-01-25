import torch
from torch import nn
import torch.nn.functional as F
from models.vgg import VGG_Backbone
import numpy as np
import torch.optim as optim
from torchvision.models import vgg16
import copy
import math
import sys

class EncoderLayers(nn.Module):
    def __init__(self, in_channel = 64, out_channel = 64, kernel_s = 3, pad = 1):
        super(EncoderLayers, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_s, stride=1, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_s, stride=1, padding=pad)
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x

class DecoderLayers(nn.Module):
    def __init__(self, in_channel = 64, out_channel = 64, kernel_s = 3, pad = 1):
        super(DecoderLayers, self).__init__()
        self.delayer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_s, stride=1, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_s, stride=1, padding=pad),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(nn.Conv2d(out_channel, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        x = self.delayer(x)
        x = self.out(x)
        return x

class AttenConsistency(nn.Module):
    def __init__(self, in_channel = 64, out_channel = 64, kernel_s = 3, pad = 1):
        super(AttenConsistency, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_s, stride=1, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_s, stride=1, padding=pad)
        )
        self.enlayer = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_s, stride=1, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_s, stride=1, padding=pad)
        )
        self.out = nn.Sequential(nn.Conv2d(out_channel, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, encoder, decoder, precoder):
        [_, _, en_H, en_W] = encoder.size()
        decoder = F.interpolate(decoder, size=(en_H, en_W), mode='bilinear', align_corners=True)

        tmp_co = encoder * decoder
        tmp_co = self.convlayer(tmp_co)

        edge = self.enlayer(precoder)

        attenC = F.interpolate(edge, size=(en_H, en_W), mode='bilinear', align_corners=True) + tmp_co

        out_edge = self.out(edge)

        return attenC, out_edge

class FICA(nn.Module):
    def __init__(self, channel = 512, reduction=16):
        super(FICA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)

        y1 = self.avg_pool(x).view(b, c)
        y1 = self.fc(y1).view(b, c, 1, 1)
        y2 = self.max_pool(x).view(b, c)
        y2 = self.fc(y2).view(b, c, 1, 1)
        cha_x = x * (y1+y2).expand_as(x)

        avg_out = torch.mean(cha_x, dim=1, keepdim=True)
        max_out, _ = torch.max(cha_x, dim=1, keepdim=True)
        cha_x = torch.cat([avg_out, max_out], dim=1)
        cha_x = self.conv(cha_x)
        cha_x = torch.where(torch.isnan(cha_x), torch.full_like(cha_x, 0), cha_x)

        return self.sigmoid(cha_x)

class AFAR(nn.Module):
    def __init__(self, channel = 20, reduction = 20):
        super(AFAR, self).__init__()

        self.avg_poolAFAR = nn.AdaptiveAvgPool2d(1)
        self.max_poolAFAR = nn.AdaptiveMaxPool2d(1)
        self.fcAFAR = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.convAFAR = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoidAFAR = nn.Sigmoid()
    
    def forward(self, x, kv = 1, EPS=sys.float_info.epsilon):
        b, c, _, _ = x.size()

        y1 = self.avg_poolAFAR(x).view(b, c)
        y1 = self.fcAFAR(y1).view(b, c, 1, 1)
        y2 = self.max_poolAFAR(x).view(b, c)
        y2 = self.fcAFAR(y2).view(b, c, 1, 1)
        y = y1 + y2

        tmp_x = []
        y = self.sigmoidAFAR(y)
        yc = y.squeeze(0).squeeze(1).squeeze(1)

        yc = (yc - torch.min(yc)) / (torch.max(yc) - torch.min(yc) + 1e-20)

        for inum in range (c):
            u, s, v = torch.svd(x[:,inum,:,:].squeeze(0))

            k = int(len(s) * kv * yc[inum])
            s_topk = torch.topk(s, k)
            u_topk = torch.index_select(u, 1, s_topk[1].squeeze())
            v_topk = torch.index_select(v, 1, s_topk[1].squeeze())
            s_topk = s_topk.values
            matrix_data = torch.mm(torch.mm(u_topk, torch.diag(s_topk)), v_topk.t())
            tmp_x.append(matrix_data.unsqueeze(0).unsqueeze(0))
        
        x = torch.cat([tmp_x[i] for i in range(len(tmp_x))], dim=1)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.convAFAR(x)
        # x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        x[(x < EPS).data] = EPS
        return self.sigmoidAFAR(x)

class CoAtten(nn.Module):
    def __init__(self):
        super(CoAtten, self).__init__()
        
        self.fica = FICA()
        self.afar = AFAR()

    def forward(self, N, co_coding2):
        for inum in range(N):
            co_coding2[inum] = self.fica(co_coding2[inum])
        co_coding2f = self.afar(torch.cat([co_coding2[i] for i in range(len(co_coding2))], dim=1))

        return co_coding2f

class IIFE(nn.Module):
    def __init__(self):
        super(IIFE, self).__init__()
        self.gradients = None
        self.backbone = VGG_Backbone()

        self.attenCon5 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))


        self.attenCon4 = AttenConsistency(in_channel=512)
        self.attenCon3 = AttenConsistency(in_channel=256)
        self.attenCon2 = AttenConsistency(in_channel=128)
        self.attenCon1 = AttenConsistency(in_channel=64)

        self.encoder4 = EncoderLayers()
        self.encoder3 = EncoderLayers()
        self.encoder2 = EncoderLayers()
        self.encoder1 = EncoderLayers()

        self.decoder4 = DecoderLayers()
        self.decoder3 = DecoderLayers()
        self.decoder2 = DecoderLayers()
        self.decoder1 = DecoderLayers()

    def set_mode(self, mode):
        self.mode = mode

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x, co_coding1, co_coding2):
        if self.mode == 'train':
            preds = self._train_forward(x, co_coding1, co_coding2)
        else:
            preds = self._test_forward(x, co_coding1, co_coding2)

        return preds
    def _test_forward(self, x, co_coding1, co_coding2):
        [_, _, H, W] = x.size()
        with torch.no_grad():
            x1 = self.backbone.conv1(x)
            x2 = self.backbone.conv2(x1)
            x3 = self.backbone.conv3(x2)
            x4 = self.backbone.conv4(x3)
        x5 = self.backbone.conv5(x4)
        x5.requires_grad_()
        x5.register_hook(self.save_gradient)
        x5_p = self.backbone.avgpool(x5)
        _x5_p = x5_p.view(x5_p.size(0), -1)
        pred_vector = self.backbone.classifier(_x5_p)

        # to 2D rec################
        coAtten = torch.cat([co_coding1[i] for i in range(len(co_coding1))], dim=0)   
        coAtten = coAtten.cuda()
        
        coFinal = torch.mean(coAtten, dim=0)
        
        similarity = torch.sum(coFinal * pred_vector)
        similarity.backward(retain_graph=True)
        cweight = F.adaptive_avg_pool2d(self.gradients, (1, 1))
        cweight = F.relu(cweight)
        cweight = (cweight - torch.min(cweight)) / (torch.max(cweight) - torch.min(cweight) + 1e-20)
        weighted_x5 = x5 * cweight

        cam = torch.mean(weighted_x5, dim=1).unsqueeze(1)
        cam = cam * co_coding2
        ########################
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-6)
        cam = torch.clamp(cam, 0, 1)

        with torch.no_grad():
            p5 = self.attenCon5(weighted_x5)
            pred5 = cam

            p4, edge4 = self.attenCon4(x4, pred5, p5)
            p4 = self.encoder4(p4)
            pred4 = self.decoder4(p4)

            p3, edge3 = self.attenCon3(x3, pred4, p4)
            p3 = self.encoder3(p3)
            pred3 = self.decoder3(p3)

            p2, edge2 = self.attenCon2(x2, pred3, p3)
            p2 = self.encoder3(p2)
            pred2 = self.decoder2(p2)

            p1, edge1 = self.attenCon1(x1, pred2, p2)
            p1 = self.encoder1(p1)
            pred1 = self.decoder1(p1)

            preds, edges = [], []
            preds.append(F.interpolate(pred5,size=(H, W),mode='bilinear',align_corners=True))
            preds.append(F.interpolate(pred4,size=(H, W),mode='bilinear',align_corners=True))
            preds.append(F.interpolate(pred3,size=(H, W),mode='bilinear',align_corners=True))
            preds.append(F.interpolate(pred2,size=(H, W),mode='bilinear',align_corners=True))
            preds.append(F.interpolate(pred1,size=(H, W),mode='bilinear',align_corners=True))
            edges.append(F.interpolate(edge4,size=(H, W),mode='bilinear',align_corners=True))
            edges.append(F.interpolate(edge3,size=(H, W),mode='bilinear',align_corners=True))
            edges.append(F.interpolate(edge2,size=(H, W),mode='bilinear',align_corners=True))
            edges.append(F.interpolate(edge1,size=(H, W),mode='bilinear',align_corners=True))

        return preds, edges
        
    
    def _train_forward(self, x, co_coding1, co_coding2):
        [_, _, H, W] = x.size()
        x1 = self.backbone.conv1(x)
        x2 = self.backbone.conv2(x1)
        x3 = self.backbone.conv3(x2)
        x4 = self.backbone.conv4(x3)
        x5 = self.backbone.conv5(x4)
        x5.register_hook(self.save_gradient)
        x5_p = self.backbone.avgpool(x5)
        _x5_p = x5_p.view(x5_p.size(0), -1)
        pred_vector = self.backbone.classifier(_x5_p)

        # to 2D rec################
        coAtten = torch.cat([co_coding1[i] for i in range(len(co_coding1))], dim=0)   
        coAtten = coAtten.cuda()
        
        coFinal = torch.mean(coAtten, dim=0)
        
        similarity = torch.sum(coFinal * pred_vector)
        similarity.backward(retain_graph=True)
        cweight = F.adaptive_avg_pool2d(self.gradients, (1, 1))
        cweight = F.relu(cweight)
        cweight = (cweight - torch.min(cweight)) / (torch.max(cweight) - torch.min(cweight) + 1e-20)
        weighted_x5 = x5 * cweight

        cam = torch.mean(weighted_x5, dim=1).unsqueeze(1)
        cam = cam * co_coding2
        ########################

        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-6)
        cam = torch.clamp(cam, 0, 1)

        p5 = self.attenCon5(weighted_x5)
        pred5 = cam

        p4, edge4 = self.attenCon4(x4, pred5, p5)
        p4 = self.encoder4(p4)
        pred4 = self.decoder4(p4)

        p3, edge3 = self.attenCon3(x3, pred4, p4)
        p3 = self.encoder3(p3)
        pred3 = self.decoder3(p3)

        p2, edge2 = self.attenCon2(x2, pred3, p3)
        p2 = self.encoder3(p2)
        pred2 = self.decoder2(p2)

        p1, edge1 = self.attenCon1(x1, pred2, p2)
        p1 = self.encoder1(p1)
        pred1 = self.decoder1(p1)

        preds, edges = [], []
        preds.append(F.interpolate(pred5,size=(H, W),mode='bilinear',align_corners=True))
        preds.append(F.interpolate(pred4,size=(H, W),mode='bilinear',align_corners=True))
        preds.append(F.interpolate(pred3,size=(H, W),mode='bilinear',align_corners=True))
        preds.append(F.interpolate(pred2,size=(H, W),mode='bilinear',align_corners=True))
        preds.append(F.interpolate(pred1,size=(H, W),mode='bilinear',align_corners=True))
        edges.append(F.interpolate(edge4,size=(H, W),mode='bilinear',align_corners=True))
        edges.append(F.interpolate(edge3,size=(H, W),mode='bilinear',align_corners=True))
        edges.append(F.interpolate(edge2,size=(H, W),mode='bilinear',align_corners=True))
        edges.append(F.interpolate(edge1,size=(H, W),mode='bilinear',align_corners=True))

        return preds, edges

class MineNet(nn.Module):
    def __init__(self, mode='train'):
        super(MineNet, self).__init__()

        self.co_classifier1 = vgg16(pretrained=True).eval()
        self.co_classifier2 = vgg16(pretrained=True).eval()
        self.co_classifier2.features = nn.Sequential(*list(self.co_classifier2.features.children())[:-1])

        self.coatten = CoAtten()
        self.iife = IIFE()
        self.mode = mode  
        
    def set_mode(self, mode):
        self.mode = mode
        self.iife.set_mode(self.mode)

    def forward(self, x):
        [_, N, _, _, _] = x.size()
        with torch.no_grad():
            ######### Co-Classify ########
            co_coding1 = []
            co_coding2 = []
            for inum in range(N):
                tmp_co_coding = self.co_classifier1(x[:, inum, :, :, :]).cpu().data.numpy() # (24, 1000)
                tmp_co_coding = torch.from_numpy(tmp_co_coding)
                tmp_co_coding = F.softmax(tmp_co_coding, dim=1)
                co_coding1.append(tmp_co_coding)
                _tmp_co_coding = self.co_classifier2.features(x[:, inum, :, :, :])
                co_coding2.append(_tmp_co_coding)

        co_coding2f = self.coatten(N, co_coding2)
        
        ########## Co-SOD ############    
        preds, edges = [], []

        for inum in range(N):
            ipreds, iedges = self.iife(x[:, inum, :, :, :],co_coding1,co_coding2f)
            preds.append(ipreds)
            edges.append(iedges)

        return preds, edges