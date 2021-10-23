# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCODataset as coco
from .penn import PennActionDataset as penn
from .bbcpose import BBCPoseDataset as bbcpose
from .bbcpose_offline import BBCPoseOfflineDataset as bbcpose_offline
from .hm36 import HM36VidDataset as hm36vid
from .hm36_offline import HM36OfflineDataset as hm36vid_offline
from .penn_offline import PennOffline as penn_offline
