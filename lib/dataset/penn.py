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
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image
import torch
from scipy.io import loadmat, savemat

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class PennActionDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None, is_ttp=False):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 13
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6)
        self.lower_body_ids = (7, 8, 9, 10, 11, 12)

        self.num_videos = 2326

        self.is_ttp = is_ttp
        self.batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
        self.ttp_batching_strategy = cfg.TRAIN.TTP_BATCHING_STRATEGY
        self.ttp_online_ref = cfg.TEST.TTP_ONLINE_REF
        if self.ttp_batching_strategy not in ['fix_one', 'both_random']:
            raise NotImplementedError

        self.selected_actions = cfg.DATASET.SELECTED_ACTIONS

        self.train_frame_frac = cfg.DATASET.TRAIN_FRAME_FRAC
        self.train_video_frac = cfg.DATASET.TRAIN_VIDEO_FRAC
        self.label_frame_frac = cfg.DATASET.LABEL_FRAME_FRAC
        self.label_video_frac = cfg.DATASET.LABEL_VIDEO_FRAC

        self.db, self.labeled_db_idx = self._get_db()
        self.length = len(self.db)  # the real dataset length used in ttp

        if is_train and cfg.DATASET.SELECT_DATA:
            assert self.train_frame_frac >= 1.0
            assert self.train_video_frac >= 1.0
            assert self.label_frame_frac >= 1.0
            assert self.label_video_frac >= 1.0
            self.db = self.select_data(self.db)
        
        if is_ttp:
            assert self.train_frame_frac >= 1.0
            assert self.train_video_frac >= 1.0
            assert self.label_frame_frac >= 1.0
            assert self.label_video_frac >= 1.0

        logger.info('=> load {} samples'.format(len(self.db)))

        self.is_imm = cfg.MODEL.IS_IMM
        self.bandwidth = cfg.DATASET.BANDWIDTH

    def _get_db(self):
        if self.is_train and (
            self.train_frame_frac <= 1.0 or 
            self.train_video_frac <= 1.0):
            # Fix the random seed if we only need part of data
            np.random.seed(0)
        
        frame_path = os.path.join(self.root, 'frames')
        label_path = os.path.join(self.root, 'labels')

        vid_idxes = list(range(self.num_videos))
        if self.is_train and self.train_video_frac <= 1.0:
            # Randomly choose self.train_video_frac videos
            vid_idxes = []
            for i in range(self.num_videos):
                label = loadmat(os.path.join(label_path, '{:04d}.mat'.format(i + 1)))
                if label['train'].item() == 1:
                    vid_idxes.append(i)
            n_vid = int(self.train_video_frac * len(vid_idxes))
            vid_idxes = np.random.choice(vid_idxes, n_vid, replace=False)
        
        labeled_vid_idxes = deepcopy(vid_idxes)
        if self.is_train and self.label_video_frac <= 1.0:
            # Randomly choose self.label_video_frac videos
            n_labeled_vid = int(self.label_video_frac * len(labeled_vid_idxes))
            labeled_vid_idxes = np.random.choice(labeled_vid_idxes, n_labeled_vid, replace=False)

        # For faster checking
        vid_idxes, labeled_vid_idxes = set(vid_idxes), set(labeled_vid_idxes)

        gt_db, labeled_db_idx = [], []
        for i in range(self.num_videos):

            label = loadmat(os.path.join(
                label_path, '{:04d}.mat'.format(i + 1)))

            if (label['train'].item() == 1) != self.is_train:
                continue
            if i not in vid_idxes:  # Randomly choose self.train_video_frac videos
                continue
            if not label['action'].item() in self.selected_actions:
                continue

            nframes = label['nframes'].item()
            start_frame = 0
            if self.is_train and self.train_frame_frac < 1.0:
                # Randomly choose continous self.train_frame_frac frames
                chosen_nframes = int(nframes * self.train_frame_frac)
                start_frame = np.random.randint(0, nframes - chosen_nframes + 1)
                nframes = chosen_nframes
            
            labeled_frame_idxes = list(range(start_frame, start_frame + nframes))
            if self.is_train and self.label_frame_frac < 1.0:
                # Randomly choose self.label_frame_frac frames
                n_labeled_frame = int(self.label_frame_frac * nframes)
                labeled_frame_idxes = np.random.choice(labeled_frame_idxes, n_labeled_frame, replace=False)
            labeled_frame_idxes = set(labeled_frame_idxes)
                
            for j in range(start_frame, start_frame + nframes):
                is_last_frame = j == start_frame + nframes - 1

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
                
                is_labeled = (i in labeled_vid_idxes and j in labeled_frame_idxes)
                if is_labeled:
                    labeled_db_idx.append(len(gt_db))

                gt_db.append(
                    {
                        'image': image_name,
                        'center': c,
                        'scale': s,
                        'joints_3d': joints_3d,
                        'joints_3d_vis': joints_3d_vis,
                        'filename': '',
                        'imgnum': 0,
                        'frame_i': j + start_frame,
                        'is_last_frame': is_last_frame,
                        'nframes': nframes,
                        'is_labeled': is_labeled,
                    }
                )
        return gt_db, labeled_db_idx


    def evaluate(self, cfg, preds, output_dir, downsample=None):
        preds = preds[:, :, 0:2]

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0
        
        threshold = 0.2

        if isinstance(downsample, int):
            gts = np.stack([data['joints_3d'][:, 0:2] for i, data in enumerate(self.db)
                        if i % downsample == downsample - 1])
            vis = np.stack([data['joints_3d_vis'][:, 0] for i, data in enumerate(self.db)
                        if i % downsample == downsample - 1])
        elif isinstance(downsample, list):
            gts = np.stack([self.db[i]['joints_3d'][:, 0:2] for i in downsample])
            vis = np.stack([self.db[i]['joints_3d_vis'][:, 0] for i in downsample])
        else:
            assert downsample is None, "Downsample should be in [int, list, None]"
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

    # for test-time-training, we multiply len(dataset) by batchsize
    # and do `idx = idx // batchsize` in `__get_item__(idx)`
    # then, when we set `shuffle=False` for dataloader,
    # each batch will have images from the same single frame
    def __len__(self):
        if self.is_ttp:
            return len(self.db) * self.batch_size
        if self.label_video_frac < 1.0 or self.label_frame_frac < 1.0:
            return len(self.labeled_db_idx)
        else:
            return len(self.db)

    def __getitem__(self, idx):
        if not self.is_ttp and (self.label_video_frac < 1.0 or self.label_frame_frac < 1.0):
            idx = self.labeled_db_idx[idx]
        if not self.is_imm:
            return super(PennActionDataset, self).__getitem__(idx)

        is_first_sample = False
        if self.is_ttp:
            is_first_sample = idx % 32 == 0
            idx = idx // 32
        
        # this method is almost the same as JointsDataset.__getitem__
        # only it returns the images of the ref_frame in addition
        # ref_frame and frame uses the same augmentation
        # it returns `(ref_input, input)` instead of `input`

        db_rec = copy.deepcopy(self.db[idx])

        # do batching
        frame_i = db_rec['frame_i']
        is_last_frame = db_rec['is_last_frame']  # is_last_frame doesn't care what frame_i really is

        # choose the reference frame and new frame_i
        if not self.is_ttp or not self.ttp_online_ref:
            # Choose reference frame from the whole video
            ref_frame_i = np.random.randint(
                max(0, frame_i - self.bandwidth), 
                min(frame_i + self.bandwidth + 1, db_rec['nframes']))
            new_frame_i = np.random.randint(
                max(0, frame_i - self.bandwidth), 
                min(frame_i + self.bandwidth + 1, db_rec['nframes']))
        else:
            # Choose reference frame from seen frames
            ref_frame_i = np.random.randint(
                max(0, frame_i - self.bandwidth), frame_i + 1)
            new_frame_i = np.random.randint(
                max(0, frame_i - self.bandwidth), frame_i + 1)
        
        if not self.is_ttp or self.ttp_batching_strategy == 'fix_one' or is_first_sample:
            new_frame_i = frame_i

        ref_db_rec = copy.deepcopy(self.db[idx - (frame_i - ref_frame_i)])
        db_rec = copy.deepcopy(self.db[idx - (frame_i - new_frame_i)])

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
        if self.is_train or (self.is_ttp and not is_first_sample):
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
                if random.random() <= 0.6 or self.is_ttp else 0

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

        input = Image.fromarray(input)
        ref_input = Image.fromarray(ref_input)
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
            'frame_i': frame_i,
            'nframes': db_rec['nframes'],
        }

        # stack to satisfy core/functions.py
        input = torch.stack([ref_input, input])
        is_labeled = db_rec['is_labeled']
        
        if self.is_train:
            return input, target, target_weight, is_labeled, meta
        else:
            return input, target, target_weight, meta
