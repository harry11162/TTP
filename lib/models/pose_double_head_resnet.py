# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from typing import no_type_check

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torch.nn.modules.container import Sequential


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, nheads=1, ffn=False):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.has_ffn = ffn
        self.attn = nn.MultiheadAttention(d_model, nheads)

        self.attn_norm = nn.LayerNorm(d_model)
        if self.has_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(inplace=True),
                nn.Linear(4 * d_model, d_model)
            )
            self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, qr, kv):
        qr_shape = qr.shape
        qr = qr.flatten(2).permute(2, 0, 1)  # ql, b, d
        kv = kv.flatten(2).permute(2, 0, 1)  # kl, b, d

        attn_ret = self.attn(qr, kv, kv, need_weights=False)[0]  # ql, b, d
        attn_ret = self.attn_norm(qr + attn_ret)

        if self.has_ffn:
            ret = self.ffn_norm(attn_ret + self.ffn(attn_ret))
        else:
            ret = attn_ret

        ret = ret.permute(1, 2, 0).reshape(*qr_shape)
        return ret

class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, d_model, nheads=1):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.nheads = nheads
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nheads, dim_feedforward=4 * d_model)
        norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, norm)
    def forward(self, qr, kv):
        qr_shape = qr.shape
        qr = qr.flatten(2).permute(2, 0, 1)  # ql, b, d
        kv = kv.flatten(2).permute(2, 0, 1)  # kl, b, d

        ret = self.decoder(qr, kv)  # ql, b, d

        ret = ret.permute(1, 2, 0).reshape(*qr_shape)

        return ret

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, height, width):
        super(PositionalEncoding, self).__init__()
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        self.d_model = d_model
        self.pe = nn.Parameter(pe)
    
    def forward(self, x):
        return x + self.pe
        

class PoseDoubleHeadResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseDoubleHeadResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        
        self.use_attn = extra.get("MHATTN", False)
        self.use_comb_attn = extra.get("COMBATTN", False)
        assert not (self.use_attn and self.use_comb_attn)
        c = extra.NUM_DECONV_FILTERS[-1]
        if self.use_attn:
            self.sup_attn, self.unsup_attn = extra.SUP_ATTN, extra.UNSUP_ATTN
            assert extra.SUP_ATTN or extra.UNSUP_ATTN
            self.sup_f = nn.Conv2d(c, c, kernel_size=1)
            self.unsup_f = nn.Conv2d(c, c, kernel_size=1)
            self.mh_attn = MultiHeadAttention(c, extra.NHEADS, extra.FFN)
            self.use_pe = extra.POS_ENC
            if self.use_pe:
                self.pe = PositionalEncoding(c, *cfg.MODEL.HEATMAP_SIZE)
        if self.use_comb_attn:
            self.sup_query = nn.Embedding(c, cfg.MODEL.NUM_JOINTS)
            if extra.get("TRANSFORMER", False):
                self.mh_attn = TransformerDecoder(extra.NUM_TF_LAYERS, c, extra.NHEADS)
            else:
                self.mh_attn = MultiHeadAttention(c, extra.NHEADS, extra.FFN)
            self.sup_weight = nn.Linear(c, cfg.MODEL.NUM_MAPS)
            self.use_pe = extra.POS_ENC
            if self.use_pe:
                self.pe = PositionalEncoding(c, *cfg.MODEL.HEATMAP_SIZE)
            

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        if not self.use_comb_attn:
            self.final_layer = nn.Conv2d(
                in_channels=extra.NUM_DECONV_FILTERS[-1],
                out_channels=cfg.MODEL.NUM_JOINTS,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            )

        self.final_layer_unsup = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_MAPS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)

        # Multi-head Attention
        if self.use_attn:
            x_sup, x_unsup = self.sup_f(x), self.unsup_f(x)
            if self.use_pe:
                x_sup_in, x_unsup_in = self.pe(x_sup), self.pe(x_unsup)
            else:
                x_sup_in, x_unsup_in = x_sup, x_unsup

            if self.sup_attn:
                x_sup_final = self.final_layer(self.mh_attn(x_sup_in, x_unsup_in))
            else:
                x_sup_final = self.final_layer(x_sup_in)

            if self.unsup_attn:
                x_unsup_final = self.final_layer_unsup(self.mh_attn(x_unsup_in, x_sup_in))
            else:
                x_unsup_final = self.final_layer_unsup(x_unsup_in)
            
            x_sup, x_unsup = x_sup_final, x_unsup_final

        elif self.use_comb_attn:
            bz, _, h, w = x.shape
            x_unsup = self.final_layer_unsup(x)  # b, 30, h, w
            sup_query = self.sup_query.weight.unsqueeze(0).repeat(bz, 1, 1)  # b, c, 7
            if self.use_pe:
                sup_weight = self.mh_attn(sup_query, self.pe(x)).transpose(1, 2)  # b, c, 7
            else:
                sup_weight = self.mh_attn(sup_query, x).transpose(1, 2)  # b, c, 7
            sup_weight = self.sup_weight(sup_weight).softmax(-1)  # b, 7, 30
            x_sup = torch.bmm(sup_weight, x_unsup.flatten(2)).reshape(bz, -1, h, w)
        
        else:
            x_sup = self.final_layer(x)
            x_unsup = self.final_layer_unsup(x)

        return x_sup, x_unsup

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            if hasattr(self, "final_layer"):
                for m in self.final_layer.modules():
                    if isinstance(m, nn.Conv2d):
                        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                        logger.info('=> init {}.bias as 0'.format(name))
                        nn.init.normal_(m.weight, std=0.001)
                        nn.init.constant_(m.bias, 0)
            if hasattr(self, "mh_attn"):
                for m in self.mh_attn.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=0.001)
                        nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


def get_double_head_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseDoubleHeadResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model