#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Sep 30 21:42:33 2021

@author: mibook

metrics
"""
import torch, pdb
from torchvision.ops import sigmoid_focal_loss

class unified(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = 0.5
        self.delta = 0.6
        self.gamma = 0.5
        self.label_smoothing = 0.1
        self.outchannels = 2

    def forward(self, pred, target):
        # Label Smoothing
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        # Masking
        mask = torch.sum(target, dim=1) == 1
        pred = pred.permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        pred = pred[mask]
        target = target[mask]
        
        # Losses
        asymmetric_ftl = self.asymmetric_focal_tversky_loss(pred, target)
        asymmetric_fl = self.asymmetric_focal_loss(pred, target)

        if self.weight is not None:
            return (self.weight * asymmetric_ftl) + ((1-self.weight) * asymmetric_fl)  
        else:
            return asymmetric_ftl + asymmetric_fl

    def asymmetric_focal_tversky_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        epsilon = 1e-07
        y_pred = y_pred.clip(epsilon, 1.0 - epsilon)

        tp = (y_true * y_pred).sum(dim=0)
        fn = (y_true * (1-y_pred)).sum(dim=0)
        fp = ((1-y_true) * y_pred).sum(dim=0)
        dice_class = (tp + epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + epsilon)
        back_dice = (1-dice_class[0]) 
        fore_dice = (1-dice_class[1]) * torch.pow(1-dice_class[1], -self.gamma) 
        return torch.stack([back_dice, fore_dice], -1).mean()

    def asymmetric_focal_loss(self, y_pred, y_true):
        epsilon = 1e-07
        y_pred = y_pred.clip(epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * torch.log(y_pred)
        back_ce = torch.pow(1 - y_pred[:,0], self.gamma) * cross_entropy[:,0]
        back_ce =  (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:,1]
        fore_ce = self.delta * fore_ce
        return torch.stack([back_ce, fore_ce], -1).sum(-1).mean()

class diceloss(torch.nn.Module):
    def __init__(self, smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        #self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        pred = pred.permute(0,2,3,1) #self.act(pred).permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        total_sum = A_sum + B_sum
        dice = 1 - ((2.0 * intersection + self.smooth) / (total_sum + self.smooth))

        dice = dice * self.w.to(device=dice.device)
        return dice.sum()

class iouloss(torch.nn.Module):
    def __init__(self, smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        #self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        pred = pred.permute(0,2,3,1) #self.act(pred).permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        union = A_sum + B_sum - intersection
        iou =  1 - ((intersection + self.smooth) / (union + self.smooth))
        iou = iou * self.w.to(device=iou.device)
        return iou.sum()

class celoss(torch.nn.Module):
    def __init__(self, smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        #self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        #pred = self.act(pred)
        ce = torch.nn.CrossEntropyLoss(weight=self.w.to(device=pred.device))(pred, torch.argmax(target, dim=1).long())
        return ce

class nllloss(torch.nn.Module):
    def __init__(self, smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        #self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        #pred = self.act(pred)
        nll = torch.nn.NLLLoss(weight=self.w.to(device=pred.device))(pred, torch.argmax(target, dim=1).long())
        return nll


class senseloss(torch.nn.Module):
    def __init__(self, smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        #self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        pred = pred.permute(0,2,3,1) #self.act(pred).permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        union = A_sum + B_sum - intersection
        iou =  -(intersection + self.smooth) / (union + self.smooth)
        iou = iou * self.w.to(device=iou.device)
        return iou.sum()


class focalloss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False, gamma=2):
        super().__init__()

    def forward(self, pred, target):
        focal_loss = sigmoid_focal_loss(pred, target, alpha = -1, gamma = 2, reduction = "mean")
        return focal_loss


class customloss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False, gamma=2):
        super().__init__()
        #self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        focal_loss = sigmoid_focal_loss(pred, target, alpha = -1, gamma = 2, reduction = "mean")
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels
        #_pred = self.act(pred)
        ce = torch.nn.CrossEntropyLoss(weight=self.w.to(device=pred.device))(pred, torch.argmax(target, dim=1).long())

        pred = pred.permute(0,2,3,1) #self.act(pred).permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        total_sum = A_sum + B_sum
        dice = 1 - ((2.0 * intersection + self.smooth) / (total_sum + self.smooth))
        dice = dice * self.w.to(device=dice.device)

        return 0.67*dice.sum() + 0.33*ce
