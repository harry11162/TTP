# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import FinetuneUnsupLoss
from core.function import test_time_training, test_time_training_offline
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True, freeze_bn=True
    )


    # Freeze layer_to_freeze and bn
    layer_to_freeze = ['final_layer', 'sup_f', 'sup_query', 'mh_attn', 'sup_weight']
    for layer in layer_to_freeze:
        if not hasattr(model.pose_net, layer):
            continue
        for p in getattr(model.pose_net, layer).parameters():
            p.requires_grad = False

    # Let's try to freeze more
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            for p in m.parameters():
                p.requires_grad = False
    
    logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
    }

    if len(cfg.GPUS) > 1:
        raise NotImplementedError
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    if not cfg.MODEL.IS_IMM:
        assert cfg.TEST.TTP_WITH_SUP, "Pose only for TTP only works with supervised frame"

    # define loss function (criterion) and optimizer
    criterion = FinetuneUnsupLoss(cfg).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        is_ttp=True,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    optimizer = get_optimizer(cfg, model)

    # use these to re-init after every video
    model_state_dict = copy.deepcopy(model.state_dict())
    optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

    if cfg.TEST.TTT_OFFLINE:
        test_time_training_offline(cfg, loader, dataset, model, model_state_dict, criterion,
                optimizer, optimizer_state_dict, final_output_dir, tb_log_dir, writer_dict)
    else:
        logger.info('test time training')
        test_time_training(cfg, loader, dataset, model, model_state_dict, criterion,
                optimizer, optimizer_state_dict, final_output_dir, tb_log_dir, writer_dict)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
