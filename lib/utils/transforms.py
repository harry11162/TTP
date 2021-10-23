# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

from PIL import ImageFilter
import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F # !!! To be removed


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
    
    def get_params(self):
        return random.uniform(self.sigma[0], self.sigma[1])

    def __call__(self, x, params=None, return_params=False):
        if params is None:
            params = self.get_params()
        sigma = params
        assert self.sigma[0] <= sigma <= self.sigma[1]
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return (x, params) if return_params else x

class RandomApply(transforms.RandomApply):
    def __init__(self, transforms, p):
        super().__init__(transforms, p=p)
    
    def forward(self, img, params=None, return_params=False):
        if params is None:
            params = (self.p < torch.rand(1), [None] * len(self.transforms))

        skip_trans, trans_params = params
        if skip_trans:
            return (img, params) if return_params else img
        
        assert len(self.transforms) == len(trans_params)
        new_trans_params = []
        for t, p in zip(self.transforms, trans_params):
            try:
                img, new_p = t(img, params=p, return_params=True)
            except TypeError:
                img, new_p = t(img), None
            
            new_trans_params.append(new_p)
            
        new_params = (skip_trans, new_trans_params)
        return (img, new_params) if return_params else img

class ColorJitter(transforms.ColorJitter):
    def __init__(self, brightness, contrast, saturation, hue):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, img, params=None, return_params=False):
        if params is None:
            params = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        
        img = params(img)
        return (img, params) if return_params else img

class RandomGrayscale(transforms.RandomGrayscale):
    def __init__(self, p):
        super().__init__(p=p)
    
    def forward(self, img, params=None, return_params=False):
        if params is None:
            params = (torch.rand(1) < self.p)
        convert_to_gray = params
        if convert_to_gray:
            num_output_channels = F._get_image_num_channels(img)
            img = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
        return (img, params) if return_params else img

class Compose(transforms.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)
    
    def __call__(self, img, params=None, return_params=False):
        if params is None:
            params = [None] * len(self.transforms)

        assert len(self.transforms) == len(params)
        new_params = []
        for t, p in zip(self.transforms, params):
            try:
                img, new_p = t(img, params=p, return_params=True)
            except TypeError:
                img, new_p = t(img), None
        
            new_params.append(new_p)

        return (img, new_params) if return_params else img
