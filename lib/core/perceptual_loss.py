# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision

class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 28):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        output = [X]
        h = self.slice1(X)
        output.append(h)
        h = self.slice2(h)
        output.append(h)
        h = self.slice3(h)
        output.append(h)
        h = self.slice4(h)
        output.append(h)
        h = self.slice5(h)
        output.append(h)
        return output


class PerceptualLoss(nn.Module):
    def __init__(self, cfg):
        super(PerceptualLoss, self).__init__()
        self.vgg16 = VGG16(requires_grad=False)
        self.vgg16.eval()
        self.criterion = nn.MSELoss(reduction='mean')
        self.loss_mean = [0.1966, 0.8725, 3.4260, 7.4396, 4.1430, 1.1304]
        self.momentum = 0.01

    def forward(self, images, pred_images):
        output_images = self.vgg16(images)
        output_pred = self.vgg16(pred_images)
        loss = []
        for i in range(len(output_images)):
            l = self.criterion(output_images[i], output_pred[i])
            l = l.mean()
            self.loss_mean[i] = self.loss_mean[i] + \
                self.momentum * (l.detach() - self.loss_mean[i])
            l = l / self.loss_mean[i]
            loss.append(l)
        loss = torch.stack(loss).sum()
        return loss