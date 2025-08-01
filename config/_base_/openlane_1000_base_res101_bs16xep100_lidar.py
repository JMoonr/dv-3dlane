import os
import os.path as osp
import numpy as np

# from lidar_utils import LidarEncodingManner

keep_n_ckpts = 2
seed = 0

dataset = '1000' # '300' | '1000'
dataset_name = 'openlane'

use_lidar = True
num_frames = 1
frame_step = 1
lidar_enc = dict(
    enc_feat_mtd=0 # LidarEncodingManner.ORIGINAL_PT_FEAT
)

train_point_pipelines = [
    dict(
        type  = 'PointsRangeFilter',
        point_cloud_range = [-74.88, -74.88, -3, 74.88, 74.88, 5]
        ),
    dict(
        type = 'PointSample',
        num_points = 16384, 
        sample_range=None), # 40.0), # TODO copy from kitt settings
    dict(
        type='PointShuffle',
    )
]

val_point_pipelines = [
    dict(
        type  = 'PointsRangeFilter',
        point_cloud_range = [-74.88, -74.88, -3, 74.88, 74.88, 5]
        ),
    dict(
        type = 'PointSample',
        num_points = 16384, 
        sample_range=None), # 40.0), # TODO copy from kitt settings
]

vis = False

output_dir = dataset_name

org_h = 1280
org_w = 1920
crop_y = 0

cam_height = 1.55
pitch = 3
fix_cam = False
pred_cam = False

no_3d = False
no_centerline = True

model_name = 'LATRMM'
mod = 'TODO'

ipm_h = 208
ipm_w = 128
resize_h = 360
resize_w = 480

encoder = 'ResNext101'
# whole model pretrained
pretrained = False
batch_norm = True

feature_channels = 128
num_proj = 4
num_att = 3
use_proj = True

max_lanes = 20

S = 72 # max sample number in img height
anchor_feat_channels = 64 # num of anchor feat channels


# top view
top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
anchor_y_steps = np.linspace(3, 103, 25)
num_y_steps = len(anchor_y_steps)

# placeholder, not used
K = np.array([[1000., 0., 960.],
            [0., 1000., 640.],
            [0., 0., 1.]])

# persformer anchor
use_default_anchor = False

batch_size = 16
nepochs = 100

no_cuda = False
nworkers = 16
seg_start_epoch = 1
start_epoch = 0
channels_in = 3

# args input
test_mode = False # 'store_true' # TODO 
evaluate = False # TODO
resume = '' # TODO
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# tensorboard
no_tb = False

# print & save
print_freq = 50
save_freq = 8

# ddp setting
dist = True
sync_bn = True
cudnn = True

distributed = True
local_rank = None #TODO
gpu = 0
world_size = 1
nodes = 1

# for reload ckpt
eval_ckpt = ''
resume_from = ''

# 2d pred setting
pred_2d = False
output_dir = 'openlane'
evaluate_case = ''
eval_freq = 8 # eval freq during training


save_prefix = osp.join(os.getcwd(), 'work_dirs')
save_path = osp.join(save_prefix, output_dir)

