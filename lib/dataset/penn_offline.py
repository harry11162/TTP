# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
import os
import json_tricks as json
from collections import OrderedDict

import cv2
import numpy as np
import torch
from scipy.io import loadmat, savemat

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class PennOffline(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train=False, transform=None, is_ttp=False):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 13
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6)
        self.lower_body_ids = (7, 8, 9, 10, 11, 12)

        self.num_videos = 2326

        self.batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)

        self.db, self.idx_range = self._get_db()
        self.cur_vid_idx = 0
        self.length = len(self.db)  # the real dataset length used in ttp

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} videos'.format(len(self.idx_range)))
        logger.info('=> load {} samples'.format(len(self.db)))

        self.is_imm = cfg.MODEL.IS_IMM
        assert self.is_imm
        self.bandwidth = cfg.DATASET.BANDWIDTH

    def _get_db(self):
        
        frame_path = os.path.join(self.root, 'frames')
        label_path = os.path.join(self.root, 'labels')

        gt_db = []
        idx_range = []
        for i in range(self.num_videos):

            label = loadmat(os.path.join(
                label_path, '{:04d}.mat'.format(i + 1)))

            if label['train'].item() == 1:
                continue

            nframes = label['nframes'].item()
            begin = len(gt_db)
            for j in range(nframes):
                is_last_frame = j == nframes - 1

                image_name = os.path.join(frame_path,
                    '{:04d}'.format(i + 1), '{:06d}.jpg'.format(j + 1))
                
                # there are mistakes in dataset, two videos are missing bbox in last frame
                if j >= label['bbox'].shape[0]:
                    bbox = label['bbox'][-1].astype(np.float)
                else:
                    bbox = label['bbox'][j].astype(np.float)
                cx = (bbox[0] + bbox[2] - 1) / 2
                cy = (bbox[1] + bbox[3] - 1) / 2
                c = np.stack([cx, cy])

                # this part is questionable
                bbox_width = bbox[2] - bbox[0] + 1
                bbox_height = bbox[3] - bbox[1] + 1
                s = max(bbox_width, bbox_height) / 200
                s = np.stack([s, s])

                # Adjust center/scale slightly to avoid cropping limbs
                if c[0] != -1:
                    c[1] = c[1] + 15 * s[1]
                s = s * 1.25

                # MPII uses matlab format, index is based 1,
                # we should first convert to 0-based index
                c = c - 1

                x = label['x'][j].astype(np.float)
                y = label['y'][j].astype(np.float)
                vis = label['visibility'][j].astype(np.float)
                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
                joints_3d[:, 0] = x[:]
                joints_3d[:, 1] = y[:]
                joints_3d_vis[:, 0] = vis[:]
                joints_3d_vis[:, 1] = vis[:]

                gt_db.append(
                    {
                        'image': image_name,
                        'center': c,
                        'scale': s,
                        'joints_3d': joints_3d,
                        'joints_3d_vis': joints_3d_vis,
                        'filename': '',
                        'imgnum': 0,
                        'frame_i': j,
                        'is_last_frame': is_last_frame,
                        'n_frames': nframes,
                    }
                )
            end = len(gt_db)
            assert end - begin == nframes
            idx_range.append((begin, end))
        return gt_db, idx_range


    def evaluate(self, cfg, preds, output_dir, downsample=None):
        preds = preds[:, :, 0:2]

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0
        
        threshold = 0.2

        if downsample is not None:
            gts = np.stack([data['joints_3d'][:, 0:2] for i, data in enumerate(self.db)
                        if i % downsample == downsample - 1])
            vis = np.stack([data['joints_3d_vis'][:, 0] for i, data in enumerate(self.db)
                        if i % downsample == downsample - 1])
        else:
            gts = np.stack([data['joints_3d'][:, 0:2] for data in self.db])
            vis = np.stack([data['joints_3d_vis'][:, 0] for data in self.db])

        error = np.linalg.norm(preds - gts, axis=2)
        neck = (gts[:, 1, :] + gts[:, 2, :]) / 2
        pelvis = (gts[:, 7, :] + gts[:, 8, :]) / 2
        torso = np.linalg.norm(neck - pelvis, axis=1)
        scaled_error = np.divide(error, torso.reshape(torso.shape[0], 1))
        vis_count = np.sum(vis, axis=0)
        less_than_threshold = np.multiply((scaled_error <= threshold), vis)

        PCK = np.divide(100.*np.sum(less_than_threshold, axis=0), vis_count)

        vis_ratio = vis_count / np.sum(vis_count).astype(np.float64)

        name_value = [
            ('Head', PCK[0]),
            ('Shoulder', 0.5 * (PCK[1] + PCK[2])),
            ('Elbow', 0.5 * (PCK[3] + PCK[4])),
            ('Wrist', 0.5 * (PCK[5] + PCK[6])),
            ('Hip', 0.5 * (PCK[7] + PCK[8])),
            ('Knee', 0.5 * (PCK[9] + PCK[10])),
            ('Ankle', 0.5 * (PCK[11] + PCK[12])),
            ('mPCK', np.sum(PCK * vis_ratio)),
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['mPCK']

    def __len__(self):
        begin, end = self.idx_range[self.cur_vid_idx]
        video_length = end - begin
        if self.is_train:
            return video_length * self.batch_size
        else:
            return video_length

    def __getitem__(self, idx):
        begin, end = self.idx_range[self.cur_vid_idx]
        if self.is_train:
            idx = np.random.randint(0, end - begin)
        idx = idx + begin

        # this method is almost the same as JointsDataset.__getitem__
        # only it returns the images of the ref_frame in addition
        # ref_frame and frame uses the same augmentation
        # it returns `(ref_input, input)` instead of `input`

        db_rec = copy.deepcopy(self.db[idx])

        # do batching
        frame_i = db_rec['frame_i']
        assert frame_i == idx - begin
        is_last_frame = db_rec['is_last_frame']  # is_last_frame doesn't care what frame_i really is

        n_frames = end - begin
        assert n_frames == db_rec['n_frames']
        ref_frame_i = np.random.randint(0, n_frames)
        ref_db_rec = copy.deepcopy(self.db[idx - (frame_i - ref_frame_i)])

        image_file = db_rec['image']
        ref_image_file = ref_db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            ref_data_numpy = zipreader.imread(
                ref_image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            ref_data_numpy = cv2.imread(
                ref_image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            ref_data_numpy = cv2.cvtColor(ref_data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))
        if ref_data_numpy is None:
            logger.error('=> fail to read {}'.format(ref_image_file))
            raise ValueError('Fail to read {}'.format(ref_image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        # if this is a test-time-training dataset,
        # we do not do augmentation for first sample,
        # which is used to calculate performance
        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                ref_data_numpy = ref_data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        ref_input = cv2.warpAffine(
            ref_data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)
            ref_input = self.transform(ref_input)
            # `self.transform` should not have any random augmentation

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'ref_image': ref_image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'is_last_frame': is_last_frame,
        }

        # stack to satisfy core/functions.py
        input = torch.stack([ref_input, input])

        return input, target, target_weight, meta