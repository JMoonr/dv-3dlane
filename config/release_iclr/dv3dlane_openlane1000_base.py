import numpy as np
from mmcv.utils import Config

_base_ = [
    '../_base_/openlane_1000_base_res101_bs16xep100_lidar.py',
    '../_base_/optimizer.py'
]
lidar_processed_dir = 'data/openlane/processed'

mod = 'release_iclr/dv3dlane_openlane1000_base'
seg_bev = True
bev_thick = 8
front_thick = 16
mask_seg_loss = False # not support for now
num_lidar_feat = 6
sample_every_n_meter = 10
sample_total_length = 50
batch_size = 8
nworkers = 10
build_multi_lidar_method = 'res_filter_maxy'

num_category = 21
pos_threshold = 0.3
top_view_region = np.array([
    [-10, 104.4], [10, 104.4], [-10, 2], [10, 2]])

# using -25.6 ~ 25.6 for PillarNet shape [%2==0] compatibility
enlarge_length_x = 15.6
zbound = [-5., 5.]
position_range = [
    top_view_region[0][0] - enlarge_length_x,
    top_view_region[2][1],
    zbound[0],
    top_view_region[1][0] + enlarge_length_x,
    top_view_region[0][1],
    zbound[1]
]
voxel_size = (0.2, 0.4, 10)
grid_size = [
    int((position_range[i + 3] - position_range[i]) / voxel_size[i])
    for i in range(3)]

anchor_y_steps = np.linspace(2, 103, 20)
pred_dim = len(anchor_y_steps)
num_y_steps = len(anchor_y_steps)

albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=1.),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.),
        ],
        p=0.5),
    dict(type='PixelDropout',
        dropout_prob=0.01,  # float
        per_channel=False,  # bool
        drop_value=0,  # ScaleFloatType | None
        mask_drop_value=None,  # ScaleFloatType | None
        always_apply=None,  # bool | None
        p=0.2,  # float
    ),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ImageCompression', quality_lower=85, quality_upper=95, p=1),
            dict(type='GaussNoise', var_limit=(10.0, 50.0), p=1),
            dict(type='RandomToneCurve',
                 scale=0.1,  # float
                 per_channel=False,  # bool
                 always_apply=None,  # bool | None
                 p=1.0,),
        ],
        p=0.2
    ),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
            dict(type='GaussianBlur', blur_limit=7, p=1.0),
            dict(type='ZoomBlur',
                 max_factor=(1, 1.05),  # ScaleFloatType
                 step_factor=(0.01, 0.01),  # ScaleFloatType
                 always_apply=None,  # bool | None
                 p=1.0),  # float)
        ],
        p=0.3),
]

dataset_cfg = dict(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225],
    dataset = '1000',
    data_label_dir = 'data/openlane/lane3d_1000/',
    data_image_dir = 'data/openlane/images/',
    train_split = 'OpenLaneSplit.Train',
    val_split = 'OpenLaneSplit.Val',
    train_pipeline = dict(
        img_aug = [
            dict(
                type='Albu',
                transforms=albu_train_transforms
            ),
        ],
        pts_aug = [
            dict(
                type = 'PointsRangeFilter',
                point_cloud_range = [-50, 0, -5, 50, 101, 5]
                ),
            dict(
                type = 'PointSample',
                num_points = 16384, 
                sample_range=None), # 40.0), # TODO copy from kitt settings
            dict(
                type='PointShuffle',
            ),
        ],
        gt3d_aug = [],
    ),
    val_pipeline = dict(
        img_aug = [],
        pts_aug = [
            dict(
                type  = 'PointsRangeFilter',
                point_cloud_range = [-50, 0, -5, 50, 101, 5]
                ),
            dict(
                type = 'PointSample',
                num_points = 16384, 
                sample_range=None), # 40.0), # TODO copy from kitt settings
            ],
        gt3d_aug = []
    ),
    seg_bev = seg_bev,
    num_lidar_feat = num_lidar_feat,
)
data_image_dir = dataset_cfg['data_image_dir']

_dim_ = 256
num_query = 40
num_pt_per_line = 20
dv3dlane_cfg = dict(
    fpn_dim = _dim_,
    num_query = num_query,
    num_group = 1,
    sparse_num_group = 4,
    pos_encoding_2d=dict(
        type='SinePositionalEncoding',
        num_feats=_dim_ // 2, normalize=True),
    q_pos_emb=dict(
        type='LiDAR_XYZ_PE',
        pos_encoding_3d = dict(
            type='Sine3D',
            embed_dims=_dim_,
            num_pt_feats=5,
            num_pos_feats=_dim_ // 2,
            temperature=10000,
            ),
        pos_emb_gen=None,
    ),
    pos_encoding_bev=None,
    encoder = dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34')
    ),
    head=dict(
        xs_loss_weight=2.0,
        zs_loss_weight=10.0,
        vis_loss_weight=1.0,
        cls_loss_weight=10,
        project_loss_weight=1.0,
        num_pt_per_line=num_pt_per_line,
        pred_dim=pred_dim,
        num_lidar_feat=num_lidar_feat - 1,
        insert_lidar_feat_before_img=True,
        neck = dict(
            type='FPN',
            in_channels=[256, 128, 256, 512],
            out_channels=_dim_,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=4,
            relu_before_extra_convs=True
        ),
        ms2one=dict(
            type='DilateNaive',
            inc=_dim_, outc=_dim_, num_scales=4,
            dilations=(1, 2, 5, 9)
        ),
        depth_net=dict(
            in_channels=256,
            mid_channels=256,
            context_channels=256,
            depth_channels=None,
            position_range=position_range,
            use_bce_loss=True,
            depth_resolution=voxel_size[1],
            norm_gt_wo_mask=False
        ),
        sparse_ins_bev=Config(
            dict(
                encoder=dict(
                    out_dims=_dim_),
                decoder=dict(
                    num_query=num_query,
                    num_group=1,
                    with_pt_center_feats=False,
                    sparse_num_group=4,
                    hidden_dim=_dim_,
                    kernel_dim=_dim_,
                    num_classes=num_category,
                    num_convs=4,
                    output_iam=True,
                    scale_factor=1., 
                    ce_weight=2.0,
                    mask_weight=5.0,
                    dice_weight=2.0,
                    objectness_weight=1.0,
                ),
                sparse_decoder_weight=5.0,
        )),
    ),
    trans_params=dict(init_z=0, bev_h=150, bev_w=70),

)

transformer=dict(
    type='DV3DLaneTransformer',
    decoder=dict(
        type='DV3DLaneTransformerDecoder',
        embed_dims=_dim_,
        num_layers=2,
        M_decay_ratio=1,
        num_query=num_query,
        num_anchor_per_query=num_pt_per_line,
        anchor_y_steps=anchor_y_steps,
        look_forward_twice=True,
        return_intermediate=False,
        transformerlayers=dict(
            type='DV3DLaneDecoderLayer',
            attn_cfgs=[
                dict(
                    type='DualModalityKMeansMHA',
                    update_query=True,
                    embed_dims=_dim_,
                    num_heads=4,
                    loss='infonce',
                    dropout=0.1),
                dict(
                    type='MSDACross3DOffset',
                    embed_dims=_dim_,
                    num_heads=4,
                    num_levels=1,
                    num_points=8,
                    batch_first=False,
                    num_query=num_query,
                    num_anchor_per_query=num_pt_per_line,
                    anchor_y_steps=anchor_y_steps,
                    voxel_size=voxel_size,
                    position_range=position_range,
                    fusion_method='se',
                    dropout=0.1),
                ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=_dim_,
                feedforward_channels=_dim_*8,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            feedforward_channels=_dim_ * 8,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                            'ffn', 'norm')),
))

sparse_ins_decoder=Config(
    dict(
        encoder=dict(
            out_dims=_dim_),
        decoder=dict(
            num_query=dv3dlane_cfg['num_query'],
            num_group=dv3dlane_cfg['num_group'],
            with_pt_center_feats=False,
            sparse_num_group=dv3dlane_cfg['sparse_num_group'],
            hidden_dim=_dim_,
            kernel_dim=_dim_,
            num_classes=num_category,
            num_convs=4,
            output_iam=True,
            scale_factor=1., 
            ce_weight=2.0,
            mask_weight=5.0,
            dice_weight=2.0,
            objectness_weight=1.0,
        ),
        sparse_decoder_weight=5.0,
))

point_backbone = dict(
    type='Pillar3DDetector',
    seg_bev=False,
    reader=dict(
        type="DynamicPFE",
        in_channels=num_lidar_feat - 1,
        num_channels=(64, ),
        pillar_size=voxel_size,
        pc_range=position_range,
    ),
    pts_backbone=dict(
        type='PillarResNet34',
        in_channels=64,
        fusion_layer=dict(
            type='Point2ImageFusion',
            img_channels=128,
            pts_channels=68,
            mid_channels=256,
            out_channels=256,
            img_levels=[0],
            align_corners=False,
            activate_out=True,
            fusion_method='only_lidar',
            fuse_out=True,
        ),
        query_layer=dict(
            type='Image2PointGridSample',
        ),
    ),
    pts_neck=dict(
        type="RPNG",
        layer_nums=[5, 5],
        num_filters=[256, 256],
        in_channels=[256, 512, 256],
    ),
    ms2one=dict(
        type='DilateNaive',
        inc=_dim_, outc=_dim_, num_scales=2,
        dilations=(1, 2)),
    bev_head=dict(
        type='SimpleBEVSegHead',
        in_channels=_dim_,
        num_classes=2,
        seg_bev_loss_weight=1.0,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce',
                loss_weight=1., reduction='mean'),
            dict(type='DiceLoss', loss_name='loss_dice',
                loss_weight=3., reduction='mean'),
        ],
        ignore_index=255,
        mask_seg_loss=False,
        sampler=None,
        align_corners=False
    ),
)

nepochs = 24
save_freq = 8
eval_freq = 8
resize_h = 720
resize_w = 960

clip_grad_norm = 20
optimizer_cfg = dict(
    type='AdamW', 
    lr=2e-4,
    betas=(0.95, 0.99),
    paramwise_cfg=dict(
        custom_keys={
            'sampling_offsets': dict(lr_mult=0.1, decay_mult=1.0),
        }),
    weight_decay=0.01)
