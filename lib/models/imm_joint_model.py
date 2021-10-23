from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pose_double_head_resnet import get_double_head_pose_net

BN_MOMENTUM = 0.1


class ImageEncoder(nn.Module):
    def __init__(self, in_channels):
        super(ImageEncoder, self).__init__()
        self.layer1 = self._make_first(in_channels, 32)
        self.layer2 = self._make_layer(32)
        self.layer3 = self._make_layer(64)
        self.layer4 = self._make_layer(128)

    def _make_first(self, in_channels, out_channels):
        conv1 = nn.Conv2d(in_channels, out_channels, 7, 1, 3, bias=False)
        bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        relu1 = nn.ReLU(inplace=True)
        
        conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        relu2 = nn.ReLU(inplace=True)

        layers = [conv1, bn1, relu1, conv2, bn2, relu2]
        return nn.Sequential(*layers)

    def _make_layer(self, in_channels):
        out_channels = in_channels * 2

        conv1 = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)
        bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        relu1 = nn.ReLU(inplace=True)
        
        conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        relu2 = nn.ReLU(inplace=True)

        layers = [conv1, bn1, relu1, conv2, bn2, relu2]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ImageRenderer(nn.Module):
    def __init__(self, in_channels):
        super(ImageRenderer, self).__init__()
        self.layer1 = self._make_layer(in_channels)
        in_channels = in_channels // 2
        self.layer2 = self._make_layer(in_channels)
        in_channels = in_channels // 2
        self.layer3 = self._make_layer(in_channels)
        in_channels = in_channels // 2
        self.layer4 = self._make_layer(in_channels, is_last=True)

    def _make_layer(self, in_channels, is_last=False):
        out_channels = in_channels // 2

        conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        relu1 = nn.ReLU(inplace=True)

        conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        relu2 = nn.ReLU(inplace=True)

        layers = [conv1, bn1, relu1, conv2, bn2, relu2]

        if not is_last:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            layers.append(upsample)
        else:
            conv3 = nn.Conv2d(out_channels, 3, 3, 1, 1, bias=True)
            layers.append(conv3)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class HumanIMMModel(nn.Module):
    def __init__(self, cfg, is_train, is_finetune, freeze_bn):
        super(HumanIMMModel, self).__init__()
        self.pose_net = get_double_head_pose_net(cfg, is_train=is_train)
        self.image_encoder = ImageEncoder(3)
        self.pose_encoder = ImageEncoder(cfg.MODEL.NUM_MAPS)
        self.image_renderer = ImageRenderer(512)
        self.pose_upsampler = nn.Upsample(scale_factor=4, mode='bilinear')

        self.sigma = cfg.MODEL.SIGMA
        self.is_finetune = is_finetune
        self.freeze_bn = freeze_bn

    def forward(self, input):
        ref_images = input[:, 0, :, :, :]
        images = input[:, 1, :, :, :]

        pose_sup, pose_unsup = self.pose_net(images)
        image_feature = self.image_encoder(ref_images)
        
        pose_feature = self.get_gaussian_map(pose_unsup)
        pose_feature = self.pose_upsampler(pose_feature)
        pose_feature = self.pose_encoder(pose_feature)
        
        feature = torch.cat([image_feature, pose_feature], dim=1)
        pred_images = self.image_renderer(feature)

        return pred_images, pose_unsup, pose_sup

    def get_gaussian_map(self, heatmaps):
        n, c, h, w = heatmaps.size()

        heatmaps_y = F.softmax(heatmaps.sum(dim=3), dim=2).reshape(n, c, h, 1)
        heatmaps_x = F.softmax(heatmaps.sum(dim=2), dim=2).reshape(n, c, 1, w)

        coord_y = heatmaps.new_tensor(range(h)).reshape(1, 1, h, 1)
        coord_x = heatmaps.new_tensor(range(w)).reshape(1, 1, 1, w)
        
        joints_y = heatmaps_y * coord_y
        joints_x = heatmaps_x * coord_x

        joints_y = joints_y.sum(dim=2)
        joints_x = joints_x.sum(dim=3)
        
        joints_y = joints_y.reshape(n, c, 1, 1)
        joints_x = joints_x.reshape(n, c, 1, 1)

        gaussian_map = torch.exp(-((coord_y - joints_y) ** 2 + (coord_x - joints_x) ** 2) / (2 * self.sigma ** 2))
        return gaussian_map

    def train(self, mode=True):
        if mode:
            super(HumanIMMModel, self).train(mode=mode)

            if self.is_finetune:
                # freeze supervise head and BNs
                self.pose_net.eval()
                self.pose_net.final_layer_unsup.train()

            if self.freeze_bn:
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        else:
            super(HumanIMMModel, self).train(mode=mode)


def get_pose_net(cfg, is_train, is_finetune=False, freeze_bn=False):
    return HumanIMMModel(cfg, is_train, is_finetune, freeze_bn)
