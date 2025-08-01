import os
import os.path as osp
import numpy as np

keep_n_ckpts = 2

dataset = '300' # '300' | '1000'
dataset_name = 'openlane'
data_dir = './data/openlane/lane3d_300/'
dataset_dir = './data/openlane/images/'
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

model_name = 'LanePETR'
mod = 'TODO'

ipm_h = 208
ipm_w = 128
resize_h = 360
resize_w = 480


encoder = 'ResNext101'
pretrained = False
batch_norm = True

feature_channels = 128
num_proj = 4
num_att = 3
use_proj = True
use_fpn = False
use_top_pathway = False

position_embedding = 'learned'
nms_thres_3d = 1.0 #meter to filter detections in BEV
new_match = False
match_dist_thre_3d = 2.0 # thresh to match an anchor to GT using new_match, meter.
max_lanes = 20
num_category = 21
y_ref = 5
prob_th = 0.5
num_class = 2 # 1 bgd | 1 lanes

S = 72 # max sample number in img height
anchor_feat_channels = 64 # num of anchor feat channels

# 2D pred settings
cls_loss_weight = 1.0 # 2D pred
reg_vis_loss_weight = 1.0 # 2D pred
nms_thres = 45.0 # nms threshold
conf_th = 0.1 # confidence thresh for selecting output 2D lanes.
vis_th = 0.1 # visibility thresh for output 2D lanes.
loss_att_weight = 100.0 # 2D lane losses weight w.r.t. 3D lane losses


# 3D loss
crit_string = 'loss_gflat'
loss_dist = [10.0, 4.0, 1.0] # vis | prob | reg

# bev seg
seg_bev = True
lane_width = 2
loss_seg_weight = 0.0
seg_start_epoch = 1


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
vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]

# tensorboard
no_tb = False

# print & save
print_freq = 50
save_freq = 50

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

