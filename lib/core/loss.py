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
import math

from .perceptual_loss import PerceptualLoss


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight, *args, **kwargs):
        is_labeled = kwargs.get('is_labeled', None)
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                joint_loss = 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                joint_loss = 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            
            # Mask unlabeled samples
            if is_labeled is not None and not is_labeled.all():
                assert False
                if is_labeled.any():
                    joint_loss = joint_loss[is_labeled].mean()
                else:
                    joint_loss = torch.tensor([0.0], requires_grad=True)
            else:
                joint_loss = joint_loss.mean()
            
            loss += joint_loss

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)


class DoubleLoss(nn.Module):
    def __init__(self, cfg):
        super(DoubleLoss, self).__init__()
        self.sup_task_loss = JointsMSELoss(cfg.LOSS.USE_TARGET_WEIGHT)
        self.unsup_task_loss = PerceptualLoss(cfg)
        self.unsup_loss_weight = cfg.LOSS.UNSUP_LOSS_WEIGHT

    def forward(self, output, target, target_weight, images, pred_images,
                joints_pred, is_labeled=None):
        sup_task_loss = self.sup_task_loss(output, target, target_weight, is_labeled=is_labeled)
        unsup_task_loss = self.unsup_task_loss(images, pred_images)
        return sup_task_loss + self.unsup_loss_weight * unsup_task_loss


class FinetuneUnsupLoss(nn.Module):
    def __init__(self, cfg):
        super(FinetuneUnsupLoss, self).__init__()
        self.unsup_task_loss = PerceptualLoss(cfg)

    def forward(self, output, target, target_weight, images, pred_images, joints_pred=None):
        unsup_task_loss = self.unsup_task_loss(images, pred_images)
        return unsup_task_loss
