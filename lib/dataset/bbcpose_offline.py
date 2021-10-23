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

logger = logging.getLogger(__name__)


class BBCPoseOfflineDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None, is_ttp=False, extra_transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 7
        self.flip_pairs = [[1, 2], [3, 4], [5, 6]]

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6)
        
        if self.is_train:
            self.video_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            self.video_idx = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]   
        self.label = loadmat(os.path.join(self.root, 'bbcpose.mat'))['bbcpose'][0]

        self.is_ttp = is_ttp
        self.batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
        self.ttp_batching_strategy = cfg.TRAIN.TTP_BATCHING_STRATEGY
        self.ttp_online_ref = cfg.TEST.TTP_ONLINE_REF
        if self.ttp_batching_strategy not in ['fix_one', 'both_random']:
            raise NotImplementedError

        self.extra_val = cfg.TEST.EXTRA_VAL
        print('get db')
        self.db, self.idx_range = self._get_db()
        self.cur_vid_idx = 0
        self.ttp_steps = cfg.TEST.TTP_STEPS
        self.val_db = [x for x in self.db if x['is_valid_frame']]
        if is_train:
            assert len(self.val_db) == 0
        elif is_ttp:
            assert len(self.val_db) + self.extra_val * len(self.video_idx) == len(self.db)
        else:
            assert len(self.val_db) == len(self.db)
        self.length = len(self.val_db)  # the real dataset length used in ttp

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

        self.is_imm = cfg.MODEL.IS_IMM
        self.bandwidth = cfg.DATASET.BANDWIDTH
        self.extra_transform = extra_transform

    def _get_db(self):
        np.random.seed(0)
        gt_db = []
        idx_range = []
        start = 0
        for i in self.video_idx:
            video_idx = i + 1
            # Pairs of (db_index, frame_num)
            train_db_idx_ele = [(idx, int(ele)) for idx, ele in enumerate(self.label[i][3][0])]
            val_db_idx_ele = [(idx, int(ele)) for idx, ele in enumerate(self.label[i][5][0])]
            val_ele = set(self.label[i][5][0])
            if self.is_train:
                db_idx_ele = copy.deepcopy(train_db_idx_ele)
            else:
                db_idx_ele = copy.deepcopy(val_db_idx_ele)
                if self.extra_val > 0:
                    # Remove duplicates
                    train_db_idx_ele = [x for x in train_db_idx_ele if x[1] not in val_ele]
                    # Remove those too far away
                    start = val_db_idx_ele[0][1] - self.extra_val // 2
                    end = val_db_idx_ele[-1][1] + self.extra_val // 2
                    train_db_idx_ele = [x for x in train_db_idx_ele if start <= x[1] <= end]
                    assert len(train_db_idx_ele) >= self.extra_val, "Not enough extra samples"
                    sampled_idx = np.random.choice(
                        np.arange(len(train_db_idx_ele)), self.extra_val, replace=False)
                    db_idx_ele += [train_db_idx_ele[idx] for idx in sampled_idx]
                    db_idx_ele.sort(key=lambda x: x[1])  # Sort by frame_num

            nframes = len(db_idx_ele)
            idx_range.append((start, start + nframes))
            start += nframes
            for frame_i, (db_index, ele) in enumerate(db_idx_ele):
                is_last_frame = (frame_i + 1 == nframes)
                image_name = os.path.join(
                    self.root, f'{video_idx}', f'{ele}.jpg')
                is_valid_frame = (ele in val_ele)
                bbox = self.__get_bbox__(i, db_index, not is_valid_frame)

                cx = (bbox[2] + bbox[0]) / 2
                cy = (bbox[3] + bbox[1]) / 2
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

                joints_3d = self.__get_joints3d(i, db_index, not is_valid_frame)
                joints_3d_vis = np.array([[1., 1., 0], [1., 1., 0], [1., 1., 0], [
                    1., 1., 0], [1., 1., 0], [1., 1., 0], [1., 1., 0]])

                gt_db.append(
                    {
                        'image': image_name,
                        'center': c,
                        'scale': s,
                        'joints_3d': joints_3d,
                        'joints_3d_vis': joints_3d_vis,
                        'filename': '',
                        'imgnum': 0,
                        'frame_i': frame_i,
                        'is_last_frame': is_last_frame,
                        'bbox': bbox,
                        'nframes': nframes,
                        'is_valid_frame': is_valid_frame,
                    }
                )

        return gt_db, idx_range

    def __get_joints3d(self, folder_id, frame_id, extension=False):
        if self.is_train or extension:
            key_index = 4
        else:
            # if self.is_ttp:
            #    key_index = 4
            # else:
            key_index = 6
        try:
            key_x = self.label[folder_id][key_index][:, :, frame_id][0, :]
            key_y = self.label[folder_id][key_index][:, :, frame_id][1, :]
            return np.array([[x, y, 0] for x, y in zip(key_x, key_y)])
        except Exception as e:
            print(folder_id, key_index, frame_id, e)

    def __get_bbox__(self, folder_id, frame_id, extension=False):
        # if self.is_train or self.is_ttp:
        if self.is_train or extension:
            key_index = 4
        else:
            key_index = 6
                
        key_x = self.label[folder_id][key_index][:, :, frame_id][0, :]
        key_y = self.label[folder_id][key_index][:, :, frame_id][1, :]
        key_xmin = min(key_x)
        key_xmax = max(key_x)
        key_ymin = min(key_y)
        key_ymax = max(key_y)
        bbox_xmin = max(0, key_xmin - 60)
        bbox_xmax = min(key_xmax + 60, 720)
        bbox_ymin = max(0, key_ymin - 60)
        bbox_ymax = min(key_ymax + 60, 480)
        return np.array([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])

    def __len__(self):
        start, end = self.idx_range[self.cur_vid_idx]
        video_length = end - start
        if self.is_train:
            return video_length * self.batch_size
        else:
            return video_length

    def evaluate(self, cfg, preds, output_dir, downsample=None):
        preds = preds[:, :, 0:2]

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        threshold_pixel = 6

        if downsample is not None:
            gts = np.stack([data['joints_3d'][:, 0:2] for i, data in enumerate(self.val_db)
                            if i % downsample == downsample - 1])
            vis = np.stack([data['joints_3d_vis'][:, 0] for i, data in enumerate(self.val_db)
                            if i % downsample == downsample - 1])
        else:
            gts = np.stack([data['joints_3d'][:, 0:2] for data in self.val_db])
            vis = np.stack([data['joints_3d_vis'][:, 0] for data in self.val_db])
        
        savemat(os.path.join(output_dir, 'gt.mat'), mdict={'gts': gts})
        preds = 0.5 * (preds-1) + 1
        gt = 0.5 * (gts-1) + 1
        err = np.sqrt(np.sum((preds - gt) ** 2, axis=2))
        #  error if regressed position is greater than 6 pixels away from ground truth
        err[err <= threshold_pixel] = 1
        err[err > threshold_pixel] = 0
        err = np.mean(err, axis=0)
        # error = np.linalg.norm(preds - gts, axis=2)
        # neck = (gts[:, 1, :] + gts[:, 2, :]) / 2
        # pelvis = (gts[:, 7, :] + gts[:, 8, :]) / 2
        # torso = np.linalg.norm(neck - pelvis, axis=1)
        # scaled_error = np.divide(error, torso.reshape(torso.shape[0], 1))
        # vis_count = np.sum(vis, axis=0)
        # less_than_threshold = np.multiply((scaled_error <= threshold), vis)

        # PCK = np.divide(100.*np.sum(less_than_threshold, axis=0), vis_count)

        # vis_ratio = vis_count / np.sum(vis_count).astype(np.float64)

        name_value = [
            ('Head', err[0]),
            ('Wrist', 0.5 * (err[1] + err[2])),
            ('Elbow', 0.5 * (err[3] + err[4])),
            ('Shoulder', 0.5 * (err[5] + err[6])),
            ('mAcc', np.mean(err)),
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['mAcc']

    # for test-time-training, we multiply len(dataset) by batchsize
    # and do `idx = idx // batchsize` in `__get_item__(idx)`
    # then, when we set `shuffle=False` for dataloader,
    # each batch will have images from the same single frame
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
        # is_last_frame doesn't care what frame_i really is
        is_last_frame = db_rec['is_last_frame']

        if 'is_valid_frame' in db_rec.keys():
            is_valid_frame = db_rec['is_valid_frame']
        else:
            is_valid_frame = True

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

        try:
            ref_db_rec = copy.deepcopy(self.db[idx - (frame_i - ref_frame_i)])
            db_rec = copy.deepcopy(self.db[idx - (frame_i - new_frame_i)])
            assert db_rec['nframes'] == ref_db_rec['nframes']
        except Exception as e:
            print(idx - (frame_i - ref_frame_i), idx, frame_i, ref_frame_i, db_rec['nframes'], len(self.db))
            print(idx - (frame_i - new_frame_i), idx, frame_i, new_frame_i, db_rec['nframes'], len(self.db))
            assert False

        image_file = db_rec['image']
        # mask_file = image_file.replace('bbc_pose', 'bbc_pose_background_removed').replace('jpg', 'png')
        ref_image_file = ref_db_rec['image']
        # ref_mask_file = ref_image_file.replace('bbc_pose', 'bbc_pose_background_removed').replace('jpg', 'png')
        bbox = db_rec['bbox']
        ref_bbox = db_rec['bbox']
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
            # mask_data_numpy = cv2.imread(
            #     mask_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            # )
            # ref_mask_data_numpy = cv2.imread(
            #     ref_mask_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            # )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            ref_data_numpy = cv2.cvtColor(ref_data_numpy, cv2.COLOR_BGR2RGB)
            # mask_data_numpy = cv2.cvtColor(mask_data_numpy, cv2.COLOR_BGR2RGB)
            # ref_mask_data_numpy = cv2.cvtColor(ref_mask_data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))
        if ref_data_numpy is None:
            logger.error('=> fail to read {}'.format(ref_image_file))
            raise ValueError('Fail to read {}'.format(ref_image_file))
        # if mask_data_numpy is None:
        #     logger.error('=> fail to read {}'.format(mask_data_numpy))
        #     raise ValueError('Fail to read {}'.format(mask_data_numpy))
        # if ref_mask_data_numpy is None:
        #     logger.error('=> fail to read {}'.format(ref_mask_data_numpy))
        #     raise ValueError('Fail to read {}'.format(ref_mask_data_numpy))            

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
        # mask_input = cv2.warpAffine(
        #     mask_data_numpy,
        #     trans,
        #     (int(self.image_size[0]), int(self.image_size[1])),
        #     flags=cv2.INTER_LINEAR)
        # ref_mask_input = cv2.warpAffine(
        #     ref_mask_data_numpy,
        #     trans,
        #     (int(self.image_size[0]), int(self.image_size[1])),
        #     flags=cv2.INTER_LINEAR)
        
        input = Image.fromarray(input)
        ref_input = Image.fromarray(ref_input)
        # mask_input = Image.fromarray(mask_input)
        # ref_mask_input = Image.fromarray(ref_mask_input)
        if self.extra_transform:
            if self.is_train or (self.is_ttp and not is_first_sample):
                input = self.extra_transform(input)
                ref_input = self.extra_transform(ref_input)
        if self.transform:
            input = self.transform(input)
            ref_input = self.transform(ref_input)
            # mask_input = self.transform(mask_input)
            # ref_mask_input = self.transform(ref_mask_input)
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
            # 'mask': mask_file,
            # 'mask_ref': ref_image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'is_last_frame': is_last_frame,
            'is_valid_frame': is_valid_frame,
        }

        # stack to satisfy core/functions.py
        # input = torch.stack([ref_input, input, ref_mask_input, mask_input])
        input = torch.stack([ref_input, input])

        return input, target, target_weight, meta