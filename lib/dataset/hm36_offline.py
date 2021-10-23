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
from PIL import Image
import torch
from scipy.io import loadmat, savemat

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

from dataset.JointsDataset import JointsDataset
import pickle as pk
import copy


logger = logging.getLogger(__name__)

LANDMARK_LABELS = {'R_Ankle': 0, 'R_Knee': 1, 'R_Hip': 2, 'L_Hip': 3, 'L_Knee': 4, 'L_Ankle': 5, 'Pelvis': 6, 'Thorax': 7,
                       'Neck': 8, 'Head': 9, 'R_Wrist': 10, 'R_Elbow': 11, 'R_Shoulder': 12, 'L_Shoulder': 13, 'L_Elbow': 14, 'L_Wrist': 15}

class HM36OfflineDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None, is_ttp=False, extra_transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.db = None
        self.data_root = root
        rect_3d_width = 2000
        rect_3d_height = 2000
        self.is_train = is_train
        self.is_ttp = is_ttp
        self.ttp_index = 0
        self.ttp_s1 = 1652
        self.ttp_s2 = 1222
        if self.is_train:
            self.pickle_cache = os.path.join(self.data_root, 'HM36Vid_trainfull_all_cache', 'HM36Vid_trainfull_all_w128xh128_keypoint_db_sample-1.pkl')
        else:
            self.pickle_cache = os.path.join(self.data_root, 'HM36Vid_validfull_all_cache', 'HM36Vid_validfull_all_w128xh128_keypoint_db_sample-1.pkl')
        
        with open(self.pickle_cache, 'rb') as fid:
            self.db_cache = pk.load(fid)
        
        self.folder_keys = list(self.db_cache.keys())
        self.num_folders = len(self.folder_keys)

        self.samples = []

        for i in range(self.num_folders):
            folder_key = self.folder_keys[i]
            folder_db = self.db_cache[folder_key]
            for j in range(len(folder_db)):
                self.samples.append((i, j))
        
        if self.is_ttp:
            self._check_keys()
        self.num_samples = len(self.samples)
        self.length = self.num_samples
        self.is_imm = cfg.MODEL.IS_IMM
        self.batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
        self.band_width = cfg.DATASET.BANDWIDTH
        self.ttp_online = cfg.TEST.TTP_ONLINE
        self.extra_transform = extra_transform
        self.rect_3d_width = rect_3d_width
        self.rect_3d_height = rect_3d_height
        self.config = {'scale_factor': 0.25,
        'rot_factor': 30,
        'color_factor': 0.2,
        'do_flip_aug': True,
        'rot_aug_rate': 0.6,
        'flip_aug_rate': 0.5}

        assert self.length == self.ttp_s1 + self.ttp_s2
        self.idx_range = [(0, self.ttp_s1), (self.ttp_s1, self.ttp_s1 + self.ttp_s2)]
        self.cur_vid_idx = 0
        self.video_idx = [0, 1]

        self.db = self._get_db()
    
    def _check_keys(self):
        cur_sub = self.folder_keys[0][0]
        ttp_length = [0]
        for k in self.folder_keys:
            if k[0] == cur_sub:
                ttp_length[-1] += len(self.db_cache[k])
            else:
                cur_sub = k[0]
                ttp_length.append(len(self.db_cache[k]))
        
        assert len(ttp_length) == 2, ttp_length
        assert ttp_length[0] == self.ttp_s1 and ttp_length[1] == self.ttp_s2, ttp_length

    def _get_db(self):
        gt_db = []
        for i in self.samples:
            folder_id, frame_id = i
            folder_key = self.folder_keys[folder_id]
            folder_db = self.db_cache[folder_key]
            the_db = folder_db[frame_id]
            image_name = the_db['image'].replace('../../data/hm36/', self.data_root)
            joints = the_db['joints_3d']
            joints_vis = the_db['joints_3d_vis']
            c = np.stack([the_db['center_x'], the_db['center_y']])
            s = max(the_db['width'], the_db['height']) / 200
            s = np.stack([s, s])
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
            s = s * 1.25
            joints[:, -1] = 0
            score = 1
            is_last_frame = frame_id == len(folder_db)-1
            gt_db.append(
                    {
                        'image': image_name,
                        'center': c,
                        'scale': s,
                        'joints_3d': joints,
                        'joints_3d_vis': joints_vis,
                        'filename': '',
                        'imgnum': 0,
                        'frame_i': frame_id,
                        'is_last_frame': is_last_frame,
                    }
                )
        self.ttp_index = 0
        return gt_db
            

    def __len__(self):
        start, end = self.idx_range[self.cur_vid_idx]
        video_length = end - start
        if self.is_train:
            return video_length * self.batch_size
        else:
            return video_length
    

    def __getitem__(self, idx):
        begin, end = self.idx_range[self.cur_vid_idx]
        if self.is_train:
            idx = np.random.randint(0, end - begin)
        idx = idx + begin

        folder_id, frame_id = self.samples[idx]
        folder_key = self.folder_keys[folder_id]
        folder_db = self.db_cache[folder_key]
        the_db = copy.deepcopy(folder_db[frame_id])
        if self.is_ttp:
            if folder_key == (11, 16, 2, 4) or folder_key == (9, 16, 2, 4):
                is_last_frame = frame_id == len(folder_db)-1
            else:
                is_last_frame = False

            if folder_key[0] == 9:
                min_id = 0
                max_id = self.ttp_s1 - 1
            elif folder_key[0] == 11:
                min_id = self.ttp_s1
                max_id = len(self.db) - 1

            next_id = random.randint(min_id, max_id)
            try:
                next_folder_id, next_frame_id = self.samples[next_id]
                next_folder_key = self.folder_keys[next_folder_id]
                next_folder_db = self.db_cache[next_folder_key]
                the_db_next = copy.deepcopy(next_folder_db[next_frame_id])
            except Exception as e:
                print(next_id)
        else:
            assert False
            min_id = max(frame_id-self.band_width, 0)
            max_id = max(frame_id-1, 0) if self.is_ttp else min(frame_id +
                                                                self.band_width, len(folder_db)-1)
            next_id = random.randint(min_id, max_id)
            is_last_frame = frame_id == len(folder_db)-1
            the_db_next = copy.deepcopy(folder_db[next_id])

        image_file = the_db['image'].replace('../../data/hm36/', self.data_root)
        ref_image_file = the_db_next['image'].replace('../../data/hm36/', self.data_root)
        data_numpy, joints, joints_vis, c, s, score = self.__get_dataset(the_db)
        ref_data_numpy, joints_next, joints_vis_next, c_next, s_next, score_next = self.__get_dataset(the_db_next)
        r = 0
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
                if random.random() <= 0.6 or self.is_ttp else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                ref_data_numpy = ref_data_numpy[:, ::-1, :]
                # mask_data_numpy = mask_data_numpy[:, ::-1, :]
                # ref_mask_data_numpy = ref_mask_data_numpy[:, ::-1, :]
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
        if self.extra_transform:
            if self.is_train or (self.is_ttp and not is_first_sample):
                input = self.extra_transform(input)
                ref_input = self.extra_transform(ref_input)
        if self.transform:
            input = self.transform(input)
            ref_input = self.transform(ref_input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'ref_image': ref_image_file,
            # 'mask': mask_file,
            # 'mask_ref': ref_image_file,
            'filename': '',
            'imgnum': 0,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'is_last_frame': is_last_frame,
        }

        input = torch.stack([ref_input, input])

        return input, target, target_weight, meta
    
    def __get_dataset(self, the_db):
        img_path = the_db['image'].replace('../../data/hm36/', self.data_root)
        data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        meta = dict()
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(img_path))
            raise ValueError('Fail to read {}'.format(img_path))
        joints = the_db['joints_3d']
        joints_vis = the_db['joints_3d_vis']
        c = np.stack([the_db['center_x'], the_db['center_y']])
        s = max(the_db['width'], the_db['height']) / 200
        s = np.stack([s, s])
        if c[0] != -1:
            c[1] = c[1] + 15 * s[1]
        s = s * 1.25
        joints[:, -1] = 0
        score = 1
        return data_numpy, joints, joints_vis, c, s, score


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
        # neck = (gts[:, 1, :] + gts[:, 2, :]) / 2
        neck = gts[:, 8, :]
        pelvis = (gts[:, 2, :] + gts[:, 3, :]) / 2
        torso = np.linalg.norm(neck - pelvis, axis=1)
        scaled_error = np.divide(error, torso.reshape(torso.shape[0], 1))
        vis_count = np.sum(vis, axis=0)
        less_than_threshold = np.multiply((scaled_error <= threshold), vis)

        PCK = np.divide(100.*np.sum(less_than_threshold, axis=0), vis_count)

        vis_ratio = vis_count / np.sum(vis_count).astype(np.float64)

        name_value = [
            ('Head', PCK[9]),
            ('Shoulder', 0.5 * (PCK[12] + PCK[13])),
            ('Elbow', 0.5 * (PCK[11] + PCK[14])),
            ('Wrist', 0.5 * (PCK[15] + PCK[10])),
            ('Hip', 0.5 * (PCK[2] + PCK[3])),
            ('Knee', 0.5 * (PCK[1] + PCK[4])),
            ('Ankle', 0.5 * (PCK[0] + PCK[5])),
            ('mPCK', np.sum(PCK * vis_ratio)),
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['mPCK']
        






        