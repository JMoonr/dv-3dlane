import re
import os
import sys
import copy
import json
import glob
import random
import pickle
import warnings
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import scipy
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from utils.utils import *
from experiments.gpu_utils import is_main_process
from data.lidar_utils import lidar2cam, cam2img, filter_fov
from data.lidar_utils import get_lidar_fov_feat, get_homo_coords
from data.lane_transform import fix_pts_interpolate, bilateral_filter_1d

from .transform import GlobalRotScaleTransImage, PhotoMetricDistortionMultiViewImage
from .data_utils import smooth_lanes
from .utils import near_one_pt, get_vis_mask


from mmdet3d.datasets.pipelines import (
        PointSample, PointShuffle, Compose, PointsRangeFilter)
from mmdet3d.core.points import LiDARPoints


sys.path.append('./')
warnings.simplefilter('ignore', np.RankWarning)
matplotlib.use('Agg')


class LaneDataset(Dataset):
    """
    Dataset with labeled lanes
        This implementation considers:
        w/o laneline 3D attributes
        w/o centerline annotations
        default considers 3D laneline, including centerlines

        This new version of data loader prepare ground-truth anchor tensor in flat ground space.
        It is assumed the dataset provides accurate visibility labels. Preparing ground-truth tensor depends on it.
    """
    # dataset_base_dir is image path, json_file_path is json file path,
    def __init__(self,
                 args, 
                 pipeline=None, 
                 logger=None,
                 split='train',
                 info_dict=None,):
        """

        :param dataset_info_file: json file list
        """
        # define image pre-processor
        self.totensor = transforms.ToTensor()

        mean = args.dataset_cfg.mean
        std  = args.dataset_cfg.std
        self.normalize = transforms.Normalize(mean, std)

        if pipeline is not None:
            img_pipes = []
            img_albu_pipe = None
            for pipe in pipeline['img_aug']:
                if pipe['type'] == 'Albu':
                    img_albu_pipe = Compose([pipe])
                else:
                    img_pipes.append(pipe)
            self.img_pipeline  = Compose(img_pipes)
            self.img_albu_pipe = img_albu_pipe
            self.pts_pipeline  = Compose(pipeline['pts_aug'])
            assert pipeline['pts_aug'][1]['type'] == 'PointSample', \
                'Please put PointSample as the second one'
            if 'pts_aug_multi_frame' in pipeline:
                assert pipeline['pts_aug_multi_frame'][1]['type'] == 'PointSample', \
                    'Please put PointSample as the second one'
                self.pts_pipeline_mf  = Compose(pipeline['pts_aug_multi_frame'])
            else:
                self.pts_pipeline_mf = self.pts_pipeline
            self.gt3d_pipeline = Compose(pipeline['gt3d_aug'])

        self.seg_bev = getattr(args, 'seg_bev', False)
        self.bev_thick = getattr(args, 'bev_thick', 2)
        self.front_thick = getattr(args, 'front_thick', 6)
        self.dataset_base_dir = args.dataset_cfg.data_image_dir
        self.json_file_path = os.path.join(args.dataset_cfg.data_label_dir + f'{split}/')
        self.lidar_processed_dir = args.lidar_processed_dir
        self.lidar_num_frames = max(1, args.num_frames)
        self.lidar_frame_step = max(1, args.frame_step)
        self.lidar_temporal_fuse_backward = getattr(args, 'lidar_temporal_fuse_backward', True)
        self.sample_every_n_meter = getattr(args, 'sample_every_n_meter', 0)
        self.sample_total_length = getattr(args, 'sample_total_length', 0)
        setattr(args, 'lidar_load_multi_frame',
                self.lidar_num_frames > 1 or self.sample_total_length > 0)

        # dataset parameters
        self.dataset_name = args.dataset_name
        self.no_3d = args.no_3d
        self.no_centerline = args.no_centerline
        self.num_category = args.num_category

        self.h_org = args.org_h
        self.w_org = args.org_w
        self.h_crop = args.crop_y

        # parameters related to service network
        self.h_net = args.resize_h
        self.w_net = args.resize_w
        
        self.top_view_region = args.top_view_region

        # LaneATT params
        self.max_lanes = args.max_lanes
        self.S = args.S
        self.n_strips = self.S - 1
        self.n_offsets = self.S
        self.strip_size = self.h_net / self.n_strips
        self.offsets_ys = np.arange(self.h_net, -1, -self.strip_size)

        self.K = args.K
        self.H_crop = homography_crop_resize(
                            [args.org_h, args.org_w], 
                            args.crop_y, 
                            [args.resize_h, args.resize_w])
        # transformation from ipm to ground region
        self.ipm_w, self.ipm_h = args.ipm_w, args.ipm_h
        self.H_ipm2g = cv2.getPerspectiveTransform(
            np.float32([[0, 0],
                        [self.ipm_w-1, 0],
                        [0, self.ipm_h-1],
                        [self.ipm_w-1, self.ipm_h-1]]),
            np.float32(args.top_view_region))
        self.H_g2ipm = np.linalg.inv(self.H_ipm2g)
        # segmentation setting
        # self.lane_width = args.lane_width

        # if args.fix_cam:
        #     self.fix_cam = True
        #     # compute the homography between image and IPM, and crop transformation
        #     self.cam_height = args.cam_height
        #     self.cam_pitch = np.pi / 180 * args.pitch
        #     self.P_g2im = projection_g2im(self.cam_pitch, self.cam_height, args.K)
        #     self.H_g2im = homograpthy_g2im(self.cam_pitch, self.cam_height, args.K)
        #     self.H_im2g = np.linalg.inv(self.H_g2im)
        #     self.H_im2ipm = np.linalg.inv(np.matmul(self.H_crop, np.matmul(self.H_g2im, self.H_ipm2g)))
        # else:
        #     self.fix_cam = False
        
        self.x_min, self.x_max = self.top_view_region[0, 0], self.top_view_region[1, 0]
        self.y_min, self.y_max = self.top_view_region[2, 1], self.top_view_region[0, 1]
        
        self.anchor_y_steps = args.anchor_y_steps
        self.num_y_steps = len(self.anchor_y_steps)
        self.anchor_y_steps_fine = np.linspace(3, 103, 40)

        self.anchor_y_steps_dense = args.get(
            'anchor_y_steps_dense',
            np.linspace(3, 103, 200))
        args.anchor_y_steps_dense = self.anchor_y_steps_dense
        self.num_y_steps_dense = len(self.anchor_y_steps_dense)
        self.anchor_dim = 3 * self.num_y_steps + args.num_category
        self.save_json_path = args.save_json_path
        
        self.use_lidar = args.use_lidar
        self.args = args
        self.logger = logger
        self.waymo_veh2gd = np.array([
                                    [0, 1, 0],
                                    [-1, 0, 0],
                                    [0, 0, 1]], dtype=float)
        label_list = glob.glob(self.json_file_path + '**/*.json', recursive=True)
        self._label_list = label_list
        self.split = split
        
        if self.use_lidar:
            # self.split = self.json_file_path.strip('/').split('/')[-1]
            if info_dict is None:
                self.logger.info(' >>> Loading Lidar infos ... | split : %s' % split)
                self.lidar_infos_dict = self.process_lidar()
            else:
                self.logger.info(" >>> Using training set `lidar_infos_dict` .")
                self.lidar_infos_dict = info_dict
        else:
            self.logger.info('No Lidar will be processed...')
            self.lidar_infos_dict = None
        
        if hasattr(self, '_label_list'):
            self.n_samples = len(self._label_list)
        else:
            self.n_samples = self._label_image_path.shape[0]

    def preprocess_data_from_json_openlane(self, idx_json_file):
        _label_image_path = None
        _label_cam_height = None
        _label_cam_pitch = None
        cam_extrinsics = None
        cam_intrinsics = None
        _label_laneline_org = None
        _label_laneline_gd = None
        _gt_laneline_visibility = None
        _gt_laneline_category_org = None

        with open(idx_json_file, 'r') as file:
            file_lines = [line for line in file]
            info_dict = json.loads(file_lines[0])

            image_path = ops.join(self.dataset_base_dir, info_dict['file_path'])
            assert ops.exists(image_path), '{:s} not exist'.format(image_path)
            _label_image_path = image_path

            T_cam2veh = np.array(info_dict['extrinsic'], dtype=np.float32)
            T_cam2gd = T_cam2veh.copy()
            T_cam2gd[:3, :3] = np.matmul(
                np.linalg.inv(self.waymo_veh2gd), T_cam2gd[:3, :3])
            T_cam2gd[0:2, 3] = 0.0

            gt_cam_height = T_cam2gd[2, 3]
            if 'cam_pitch' in info_dict:
                gt_cam_pitch = info_dict['cam_pitch']
            else:
                gt_cam_pitch = 0

            cam_intrinsics = np.array(info_dict['intrinsic'])

            _label_cam_height = gt_cam_height
            _label_cam_pitch = gt_cam_pitch

            gt_lanes_packed_cam = info_dict['lane_lines']
            gt_lane_pts, gt_lane_visibility, gt_laneline_category = [], [], []
            for i, gt_lane_packed in enumerate(gt_lanes_packed_cam):
                # A GT lane can be either 2D or 3D
                # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                lane = np.array(gt_lane_packed['xyz'], dtype=np.float32)
                lane_visibility = np.array(gt_lane_packed['visibility'])

                # Coordinate convertion for openlane_300 data
                lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
                lane = np.matmul(T_cam2gd,  lane)

                lane = lane[0:3, :].T
                gt_lane_pts.append(lane)
                gt_lane_visibility.append(lane_visibility)

                if 'category' in gt_lane_packed:
                    lane_cate = gt_lane_packed['category']
                    if lane_cate == 21:  # merge left and right road edge into road edge
                        lane_cate = 20
                    gt_laneline_category.append(lane_cate)
                else:
                    gt_laneline_category.append(1)
        
        _gt_laneline_category_org = copy.deepcopy(np.array(gt_laneline_category))

        _label_laneline_gd = gt_lane_pts
        gt_visibility = gt_lane_visibility

        _label_laneline_gd = [prune_3d_lane_by_visibility(gt_lane_gd, gt_visibility[k]) for k, gt_lane_gd in enumerate(_label_laneline_gd)]
        _label_laneline_gd = [lane[lane[:,1].argsort()] for lane in _label_laneline_gd]
        T_cam2gd = T_cam2gd @ np.array(
            [[0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]], dtype=float)

        processed_info_dict = {
           'label_image_path': _label_image_path,
           'label_cam_height': _label_cam_height,
           'label_cam_pitch' : _label_cam_pitch,
           'cam2gd'  : T_cam2gd,
           'cam2veh' : T_cam2veh,
           'cam_intrinsics' : cam_intrinsics,
           'label_laneline_gd' : _label_laneline_gd,
           'label_lanline_org' : gt_lanes_packed_cam,
           'gt_laneline_visibility' : _gt_laneline_visibility,
           'gt_laneline_category_org' : _gt_laneline_category_org
        }
        return processed_info_dict, info_dict

    def process_lidar(self):
        # split = 'train' if 'train' in self.json_file_path else 'val'
        split = 'all'
        lidar_info_cache_path = f'.cache/infos_dict_{split}.pkl'

        if not os.path.exists(self.lidar_processed_dir):
            self.logger.error("waymo data path doesn't EXIST ! please check the '--lidar_processed_dir' input value: %s" % self.lidar_processed_dir)
            raise ValueError(f'{self.lidar_processed_dir} not exist.')

        lidar_infos_dict = None
        self.logger.info("[INFO] Load lidar infos...")

        if os.path.isfile(lidar_info_cache_path):
            self.logger.info(' >>> load lidar info_dict from .cache ...')
            with open(lidar_info_cache_path, 'rb') as f:
                lidar_infos_dict = pickle.load(f)
        else:
            self.logger.info(' >>> going to parse lidar info ...')
            lidar_infos_dict = defaultdict(dict)
            train_annos_dir = f'{self.lidar_processed_dir}/training/annos'
            val_annos_dir = f'{self.lidar_processed_dir}/validation/annos'

            train_info_tokens = os.listdir(train_annos_dir)
            train_info_tokens = [f'{train_annos_dir}/{_t}' for _t in train_info_tokens]

            val_info_tokens = os.listdir(val_annos_dir)
            val_info_tokens = [f'{val_annos_dir}/{_t}' for _t in val_info_tokens]

            all_info_tokens = train_info_tokens + val_info_tokens

            for info_token in tqdm(all_info_tokens):
                anno_pkl = info_token
                info_token = ops.basename(info_token)
                with open(anno_pkl, 'rb') as f:
                    annos = pickle.load(f)
                frame_name = annos['frame_name'].split('_')
                timestamp = frame_name[-1]
                scene_name = annos['scene_name']
                frame_key = timestamp[:12]
                seg_name_key = 'segment-' + scene_name + '_with_camera_labels'
                assert frame_key not in lidar_infos_dict[seg_name_key], 'frame_key already in lidar_infos_dict'
                lidar_infos_dict[seg_name_key][frame_key] = dict(
                    token=info_token,
                    lidar_root=ops.dirname(anno_pkl).replace('annos', 'lidar')
                )
            with open(lidar_info_cache_path, 'wb') as f:
                pickle.dump(lidar_infos_dict, f)

        def _read_extr(info):
            if 'extr' in info:
                return info['extr'].reshape(4, 4)
            if hasattr(self.args, 'lidar_root'):
                lidar_root = os.path.join(
                    self.args.lidar_root,
                    '/'.join(info['lidar_root'].split('/')[-2:]))
            else:
                lidar_root = info['lidar_root']
            token_file = ops.join(lidar_root[:-len('lidar')], 'annos', info['token'])
            with open(token_file, 'rb') as f:
                annos_info = pickle.load(f)
                extr = annos_info['veh_to_global']
                info['extr'] = extr
            return extr.reshape(4, 4)

        if self.sample_every_n_meter > 0:
            assert self.lidar_num_frames == 1, 'can not use both'
            # lidar_adj_info_cache_spec = f'.cache/infos_dict_adj_{self.sample_every_n_meter}m_{self.sample_total_length}m_{split}.pkl'
            lidar_adj_info_cache_spec = f'.cache/infos_dict_adj_{self.sample_every_n_meter}m_{self.sample_total_length}m_all.pkl'
            lidar_adj_info_cache = f'.cache/infos_dict_all_with_extr.pkl'

            if os.path.isfile(lidar_adj_info_cache_spec):
                self.logger.info(f' >>> load lidar adj frames info_dict from {lidar_adj_info_cache_spec} ...')
                with open(lidar_adj_info_cache_spec, 'rb') as f:
                    lidar_infos_dict = pickle.load(f)
            else:
                if os.path.isfile(lidar_adj_info_cache):
                    self.logger.info(f' >>> load lidar adj frames info_dict from {lidar_adj_info_cache} ...')
                    with open(lidar_adj_info_cache, 'rb') as f:
                        lidar_infos_dict = pickle.load(f)
                else:
                    self.logger.info("Read all extr....")
                    for scene_name, frames in tqdm(lidar_infos_dict.items()):
                        for frame_idx, info in frames.items():
                            _read_extr(info)
                    with open(lidar_adj_info_cache, 'wb') as f:
                        pickle.dump(lidar_infos_dict, f)

                self.logger.info("Build adj frames....")
                for scene_name, frames in tqdm(lidar_infos_dict.items()):
                    all_frames = list(frames.keys())
                    sorted_all_frames = list(np.sort(all_frames))
                    for frame_idx, info in frames.items():
                        cur_idx = sorted_all_frames.index(frame_idx)
                        first_extr = cur_extr = info['extr'].reshape(4, 4) # _read_extr(info)

                        adj_frames = []
                        for next_frame in sorted_all_frames[cur_idx + 1:]:
                            next_info = frames[next_frame]
                            next_extr = next_info['extr'].reshape(4, 4) # _read_extr(next_info)
                            rel_trans = np.linalg.inv(next_extr) @ cur_extr
                            delta_x = np.linalg.norm(rel_trans[0:2, -1])

                            if delta_x > self.sample_every_n_meter:
                                adj_frames.append(next_frame)
                                cur_extr = next_extr

                                if np.linalg.norm((np.linalg.inv(next_extr) @ first_extr)[0:2, -1]) \
                                        > self.sample_total_length:
                                    break
                        info['adj_frames'] = adj_frames
                with open(lidar_adj_info_cache_spec, 'wb') as f:
                    pickle.dump(lidar_infos_dict, f)
        else:
            for scene_name, frames in tqdm(lidar_infos_dict.items()):
                all_frames = list(frames.keys())
                sorted_all_frames = list(np.sort(all_frames))

                for frame_idx, v in frames.items():
                    cur_idx = sorted_all_frames.index(frame_idx)
                    adj_frames = []
                    for frame_offset in range(1, self.lidar_num_frames // self.lidar_frame_step):
                        if self.lidar_temporal_fuse_backward:
                            target_idx = cur_idx - frame_offset * self.args.frame_step
                        else:
                            target_idx = cur_idx + frame_offset * self.args.frame_step
                        if target_idx < 0 or target_idx >= len(sorted_all_frames):
                            adj_frames.append(False)
                        else:
                            adj_frames.append(sorted_all_frames[target_idx])
                    v['adj_frames'] = adj_frames
        return lidar_infos_dict

    def get_lidar_pts(self, idx_json_file, idx):
        '''
        return:
            points_all : 5-d points, if multi-frame is used, adj_frames are aligned to cur_frame via global2cur_cam @ veh2gobal.
        '''
        points_all = None
        if not self.use_lidar:
            return points_all

        if self.lidar_infos_dict is not None:
            path_parts = idx_json_file.split('/')
            scene_name = path_parts[-2]
            timestamp_name = path_parts[-1].split('.')[0]
            split = path_parts[-3]

            if scene_name in self.lidar_infos_dict:
                timestamp_name = timestamp_name[:12]
                info = self.lidar_infos_dict[scene_name][timestamp_name]
                infos = [info]

                for _frame in info['adj_frames']:
                    if _frame:
                        infos.append(self.lidar_infos_dict[scene_name][_frame])
                    else:
                        infos.append(None)
                points_all_list = []
                for i, info in enumerate(infos):
                    if not info:
                        break
                    if hasattr(self.args, 'lidar_root'):
                        lidar_root = os.path.join(
                            self.args.lidar_root,
                            '/'.join(info['lidar_root'].split('/')[-2:]))
                    else:
                        lidar_root = info['lidar_root']
                    lidar_file = ops.join(lidar_root, info['token'])
                    token_file = ops.join(lidar_root[:-len('lidar')], 'annos', info['token'])
                    with open(lidar_file, 'rb') as f:
                        lidar_info = pickle.load(f)
                    with open(token_file, 'rb') as f:
                        annos_info = pickle.load(f)
                        extr = annos_info['veh_to_global']
                    info['extr'] = extr.reshape(4, 4)

                    lidars = lidar_info['lidars']
                    if i != 0:
                        aligned_points = np.linalg.inv(infos[0]['extr']) \
                            @ info['extr'] \
                            @ get_homo_coords(lidars['points_xyz'], with_return_T=True)
                        lidars['points_xyz'] = aligned_points.T[:, :3]
                    points_all = np.concatenate(
                        [lidars['points_xyz'],
                         lidars['points_feature'],
                         np.ones((lidars['points_xyz'].shape[0], 1)) * i], axis=1
                    )
                    points_all_list.append(points_all)
                points_all = np.concatenate(points_all_list, axis=0)
            else:
                points_all = None
        return points_all

    def __len__(self):
        """
        Conventional len method
        """
        return self.n_samples

    def process_points(self, 
                       points_all=None, 
                       T_cam2veh=None,
                       T_cam2gd=None,
                       intrinsics=None,
                       image=None
                       ):
        extra_dict = {}
        if self.use_lidar and points_all is not None:
            # lidar_pt = points_all.copy()
            R_vg = np.array([[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1]], dtype=float)
            R_gc = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]], dtype=float)
            T_norm_cam2veh = T_cam2veh.copy()
            T_norm_cam2veh[:3, :3] = T_cam2veh[:3, :3] @ R_vg @ R_gc
            norm_cam_pt = lidar2cam(points_all, T_norm_cam2veh)
            img_pt, depth = cam2img(norm_cam_pt, intrinsics)
            img_pt, mask = filter_fov(
                img_pt, depth, image.height, image.width)

            # TODO lidar_encoding_manner: pt / voxel (pillar)
            lidar_fov = norm_cam_pt[mask].copy()
            lidar_fov_xyz, lidar_fov_feat = lidar_fov[:, :3], lidar_fov[:, 3:]
            lidar_fov_xyz = (
                T_cam2gd @ get_homo_coords(lidar_fov_xyz, with_return_T=True)).T

            lidar_fov_feat = get_lidar_fov_feat(
                enc_feat_mtd=self.args.lidar_enc['enc_feat_mtd'],
                lidar_fov_feat_org=lidar_fov_feat,
                image=image,
                img_pt=img_pt
            )
            lidar_fov = np.concatenate([lidar_fov_xyz[:, :3], lidar_fov_feat], axis=1)

            extra_dict['points'] = lidar_fov
        return extra_dict

    def gen_M(self, T_cam2gd, intrinsics, processed_info_dict, aug_mat=None):
        H_g2im, P_g2im, H_crop, H_im2ipm = self.transform_mats_impl(
            T_cam2gd,
            intrinsics, 
            processed_info_dict['label_cam_pitch'], 
            processed_info_dict['label_cam_height']
        )
        M = np.matmul(H_crop, P_g2im)
        # update transformation with image augmentation
        if aug_mat is not None:
            M = np.matmul(aug_mat, M)
        
        results = dict(
            H_g2im = H_g2im,
            P_g2im = P_g2im,
            H_crop = H_crop,
            H_im2ipm = H_im2ipm,
            M = M
        )
        return results

    def gen_seg_labels(self, 
                       T_gd_label2img, 
                       gt_lanes_gd, 
                       gt_lanes_category, 
                       front_thick=6,
                       bev_thick=2,
                       points=None,
                       ):
        out = {}
        # prepare binary segmentation label map
        seg_label = np.zeros((self.h_net, self.w_net), dtype=np.int8)
        # seg idx has the same order as gt_lanes
        seg_idx_label = np.zeros((self.max_lanes, self.h_net, self.w_net), dtype=np.uint8)
        
        ground_lanes = np.zeros((self.max_lanes, self.anchor_dim), dtype=np.float32)
        ground_lanes_dense = np.zeros(
            (self.max_lanes, self.num_y_steps_dense * 3), dtype=np.float32)
        gt_laneline_img = [[0]] * len(gt_lanes_gd)

        if self.seg_bev:
            bev_seg_label = np.zeros(
                (self.args.grid_size[1], self.args.grid_size[0]),
                dtype=np.uint8)
            bev_seg_idx = np.zeros(
                (self.max_lanes, self.args.grid_size[1], self.args.grid_size[0]),
                dtype=np.uint8)
            assert points is not None
            if self.args.lidar_load_multi_frame:
                single_points = points[points[..., -1] == 0]
            else:
                single_points = points
            max_v = single_points.max(0)[0]
            min_v = single_points.min(0)[0]
            max_x, max_y = max_v[:2]
            min_x, min_y = min_v[:2]
            max_x_bev = (max_x - self.args.position_range[0]) / self.args.voxel_size[0]
            min_x_bev = (min_x - self.args.position_range[0]) / self.args.voxel_size[0]
            max_y_bev = (max_y - self.args.position_range[1]) / self.args.voxel_size[1]
            min_y_bev = (min_y - self.args.position_range[1]) / self.args.voxel_size[1]
            bev_seg_mask = np.zeros_like(bev_seg_label)
            bev_seg_mask[int(min_y_bev) : int(max_y_bev), int(min_x_bev) : int(max_x_bev)] = 1
            out['bev_seg_mask'] = bev_seg_mask

        seg_interp_label = seg_label.copy()
        for i, lane in enumerate(gt_lanes_gd):
            lane = lane.astype(np.float32)
            if i >= self.max_lanes:
                break

            # TODO remove this
            if lane.shape[0] <= 2:
                continue

            if gt_lanes_category[i] >= self.num_category:
                continue

            vis = get_vis_mask(self.anchor_y_steps, lane, tol_dist=5)
            xs, zs = resample_laneline_in_y(
                lane, self.anchor_y_steps,
                interp_kind='linear',
                outrange_use_polyfit=False)

            lane = np.stack([xs, self.anchor_y_steps, zs], axis=1)

            if vis.sum() < 2:
                continue
            
            x_2d, y_2d = projective_transformation(
                T_gd_label2img[:3],
                xs[vis], # xs_dense[vis_dense],
                self.anchor_y_steps[vis], # self.anchor_y_steps_dense[vis_dense],
                zs[vis], # zs_dense[vis_dense]
            )
            lane2d = np.stack([x_2d, y_2d], axis=-1)
            dec_idx = np.argsort(lane2d[:, 1])[::-1]
            lane2d = lane2d[dec_idx]
            if lane2d.shape[0] == 1 and near_one_pt(lane2d):
                seg_label = cv2.circle(seg_label, 
                                       (int(x_2d[0]), int(y_2d[0])), # tuple(map(np.int32, lane2d)),
                                       front_thick,
                                       1, # gt_lanes_category[i].item(),
                                       -1)
                seg_idx_label[i] = cv2.circle(seg_idx_label[i], 
                                              (int(x_2d[0]), int(y_2d[0])), # tuple(map(np.int32, lane2d)), 
                                              front_thick,
                                              gt_lanes_category[i].item(),
                                              -1)
            else:
                seg_label = cv2.polylines(
                    seg_label,
                    [np.int32(lane2d).reshape((-1, 1, 2))],
                    isClosed=False,
                    color=1,
                    thickness=front_thick
                )
                seg_idx_label[i] = cv2.polylines(
                    seg_idx_label[i],
                    [np.int32(lane2d).reshape((-1, 1, 2))],
                    isClosed=False,
                    color=gt_lanes_category[i].item(),
                    thickness=front_thick
                )
            
            if seg_idx_label[i].max() <= 0:
                continue
            
            if self.seg_bev:
                xs_bev = (xs - self.args.position_range[0]) / self.args.voxel_size[0]
                ys_bev = (self.anchor_y_steps - self.args.position_range[1]) / self.args.voxel_size[1]
                xs_bev_pos = xs_bev[vis]
                ys_bev_pos = ys_bev[vis]
                xs_bev_pos, ys_bev_pos = smooth_lanes(xs_bev_pos, ys_bev_pos)
                
                lane2d_bev = np.stack([xs_bev_pos, ys_bev_pos], axis=-1).astype(np.int32)
                bev_seg_label = cv2.polylines(bev_seg_label, [lane2d_bev], 
                                              isClosed=False, 
                                              color=gt_lanes_category[i].item(), thickness=bev_thick)
                bev_seg_idx[i] = cv2.polylines(bev_seg_idx[i], 
                                               [lane2d_bev], 
                                               isClosed=False, 
                                               color=gt_lanes_category[i].item(), thickness=bev_thick)
                
                if seg_idx_label[i].max() != bev_seg_idx[i].max():
                    seg_idx_label[i][:] = 0
                    bev_seg_idx[i][:] = 0
                    continue

            if bev_seg_label.max() <= 0:
                continue

            ground_lanes[i][0: self.num_y_steps] = xs
            ground_lanes[i][self.num_y_steps:2*self.num_y_steps] = zs
            ground_lanes[i][2*self.num_y_steps:3*self.num_y_steps] = vis * 1.0
            ground_lanes[i][self.anchor_dim - self.num_category] = 0.0
            ground_lanes[i][self.anchor_dim - self.num_category + gt_lanes_category[i]] = 1.0

            xs_dense, zs_dense = resample_laneline_in_y(
                lane, self.anchor_y_steps_dense,
                outrange_use_polyfit=True)
            vis_dense = np.logical_and(
                self.anchor_y_steps_dense > lane[:, 1].min(),
                self.anchor_y_steps_dense < lane[:, 1].max())
            ground_lanes_dense[i][0: self.num_y_steps_dense] = xs_dense
            ground_lanes_dense[i][1*self.num_y_steps_dense: 2*self.num_y_steps_dense] = zs_dense
            ground_lanes_dense[i][2*self.num_y_steps_dense: 3*self.num_y_steps_dense] = vis_dense * 1.0

        out.update(dict(
            seg_label = seg_label,
            seg_idx_label = seg_idx_label,
            bev_seg_label = bev_seg_label,
            ground_lanes = ground_lanes,
            ground_lanes_dense = ground_lanes_dense,
            bev_seg_idx=bev_seg_idx,
        ))
        return out

    # new getitem, WIP
    def WIP__getitem__(self, idx):
        """
        Args: idx (int): Index in list to load image
        """
        idx_json_file = self._label_list[idx]
        points_all = self.get_lidar_pts(idx_json_file=idx_json_file, idx=idx)
        # preprocess data from json file
        processed_info_dict, json_info_dict = self.preprocess_data_from_json_openlane(idx_json_file)
        
        gt_cam_height = processed_info_dict['label_cam_height']
        gt_cam_pitch = processed_info_dict['label_cam_pitch']
            
        intrinsics = processed_info_dict['cam_intrinsics']
        T_cam2gd = processed_info_dict['cam2gd']
        T_cam2veh = processed_info_dict['cam2veh']
        img_name = processed_info_dict['label_image_path']
       
        with open(img_name, 'rb') as f:
            image = (Image.open(f).convert('RGB'))

        # TODO extra_points
        extra_dict = self.process_points(points_all=points_all,
                                         T_cam2veh=T_cam2veh,
                                         T_cam2gd=T_cam2gd,
                                         intrinsics=intrinsics,
                                         image=image)

        # image preprocess with crop and resize
        image = F.crop(image, self.h_crop, 0, self.h_org-self.h_crop, self.w_org)
        image = F.resize(
            image, size=(self.h_net, self.w_net),
            interpolation=InterpolationMode.BILINEAR)
        aug_mat = np.eye(3)
        if hasattr(self, 'img_pipeline'):
            if self.img_albu_pipe is not None:
                image = self.img_albu_pipe(dict(img=np.array(image)))['img']

            aug_dict = self.img_pipeline(dict(img=np.array(image)))
            image = Image.fromarray(
                np.clip(aug_dict['img'], 0, 255).astype(np.uint8))
            aug_mat = aug_dict.get('rot_mat', np.eye(3))

        trans = self.gen_M(T_cam2gd, intrinsics, processed_info_dict, aug_mat)
        T_gd_label2img = np.eye(4).astype(np.float32)
        T_gd_label2img[:3] = trans['M']

        if hasattr(self, 'gt3d_pipeline'):
            results = self.gt3d_pipeline(
                dict(
                    lidar2img=T_gd_label2img,
                    gt_lanes =processed_info_dict['label_laneline_gd'],
                    points   = extra_dict['points'] if self.use_lidar else None,
                    H=image.height, W=image.width,
                )
            )
            T_gd_label2img = results['lidar2img']
            processed_info_dict['label_laneline_gd'] = results['gt_lanes']

            if 'pT' in results:
                perspective_trans = results['pT']
                np_image = np.array(image).astype(np.float32)
                image = cv2.warpPerspective(
                    np_image,
                    perspective_trans,
                    dsize=(image.width, image.height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0).astype(np.uint8)
                image = Image.fromarray(image)
            if self.use_lidar:
                extra_dict['points'] = results['points']

        lidar_fov = extra_dict['points']
        if self.args.lidar_load_multi_frame:
            lidar_fov_sf = lidar_fov[..., :-1][lidar_fov[..., -1] == 0]
            # put pipeline here on purpose, make kd simple
            lidar_fov_sf = self.pts_pipeline(
                dict(points=LiDARPoints(lidar_fov_sf, points_dim=lidar_fov.shape[1] - 1))
            )['points'].tensor

            build_multi_lidar_method = getattr(self.args, 'build_multi_lidar_method', None)
            if build_multi_lidar_method is None:
                lidar_fov_mf = lidar_fov[..., :-1][lidar_fov[..., -1] > 0]
                if lidar_fov_mf.shape[0]:
                    lidar_fov_mf = self.pts_pipeline_mf(
                        dict(points=LiDARPoints(lidar_fov_mf, points_dim=lidar_fov.shape[1] - 1))
                    )['points'].tensor
                else:
                    # dummy for empty multi-frame
                    # TODO hardcode, assume the second one is PointSample
                    mf_num_points = self.pts_pipeline_mf.transforms[1].num_points
                    lidar_fov_mf = lidar_fov_sf[
                        np.random.choice(np.arange(lidar_fov_sf.shape[0]), mf_num_points)]
            elif build_multi_lidar_method == 'res_filter_maxy':
                lidar_fov_mf_all = lidar_fov[..., :-1][lidar_fov[..., -1] > 0]
                res_lidar = lidar_fov_sf.clone()
                maxx, maxy, maxz = lidar_fov_sf.max(0)[0][:3]
                flag = False
                if lidar_fov_mf_all.shape[0]:
                    lidar_fov_mf_filter = lidar_fov_mf_all[lidar_fov_mf_all[..., 1] >= maxy.item()]
                    if lidar_fov_mf_filter.shape[0] > 10:
                        try:
                            lidar_fov_mf = self.pts_pipeline_mf(
                                dict(points=LiDARPoints(
                                    lidar_fov_mf_filter, points_dim=lidar_fov_mf_filter.shape[1]))
                            )['points'].tensor
                            lidar_fov_mf = torch.cat([res_lidar, lidar_fov_mf], dim=0)
                            flag = True
                        except ValueError:
                            flag = False
                if not flag:
                    mf_num_points = self.pts_pipeline_mf.transforms[1].num_points
                    lidar_fov_mf = lidar_fov_sf[
                        np.random.choice(np.arange(lidar_fov_sf.shape[0]), mf_num_points)]
                    lidar_fov_mf = torch.cat([res_lidar, lidar_fov_mf], dim=0)
            else:
                assert False

            lidar_fov = torch.cat([
                torch.cat([lidar_fov_sf, torch.zeros((lidar_fov_sf.shape[0], 1))], dim=-1),
                torch.cat([lidar_fov_mf, torch.ones((lidar_fov_mf.shape[0], 1))], dim=-1),
            ], dim=0)
        else:
            lidar_fov = self.pts_pipeline(
                dict(points=LiDARPoints(lidar_fov, points_dim=lidar_fov.shape[1])))
            lidar_fov = lidar_fov['points'].tensor
        extra_dict['points'] = lidar_fov[:, :self.args.num_lidar_feat]

        seg_labels = self.gen_seg_labels(
            T_gd_label2img, 
            processed_info_dict['label_laneline_gd'],
            processed_info_dict['gt_laneline_category_org'],
            front_thick=self.front_thick,
            bev_thick=self.bev_thick,
            points=extra_dict['points'] if self.use_lidar else None
        )

        image = self.totensor(image).float()
        image = self.normalize(image)
        intrinsics = torch.from_numpy(intrinsics).float()
        T_cam2gd = torch.from_numpy(T_cam2gd).float()
        T_cam2veh = torch.from_numpy(T_cam2veh).float()

        seg_label = torch.from_numpy(seg_labels['seg_label'].astype(np.float32))
        seg_label.unsqueeze_(0)

        extra_dict['seg'] = seg_label
        extra_dict['lane_idx'] = seg_labels['seg_idx_label']
        extra_dict['ground_lanes'] = seg_labels['ground_lanes']
        extra_dict['ground_lanes_dense'] = seg_labels['ground_lanes_dense']
        extra_dict['lidar2img'] = T_gd_label2img
        extra_dict['pad_shape'] = torch.Tensor(seg_labels['seg_idx_label'].shape[-2:]).float()
        extra_dict['idx_json_file'] = idx_json_file
        extra_dict['image'] = image
        extra_dict['intrinsics'] = intrinsics

        if hasattr(self, 'img_pipeline'):
            aug_mat = torch.from_numpy(aug_mat.astype(np.float32))
            extra_dict['aug_mat'] = aug_mat
        if self.seg_bev:
            extra_dict['bev_seg_idx_label'] = seg_labels['bev_seg_label']
            extra_dict['bev_seg_mask'] = seg_labels['bev_seg_mask']
            extra_dict['bev_seg_idx'] = seg_labels['bev_seg_idx']
        return extra_dict

    # old getitem, workable
    def __getitem__(self, idx):
        """
        Args: idx (int): Index in list to load image
        """
        return self.WIP__getitem__(idx)

    def transform_mats(self, idx):
        """
            return the transform matrices associated with sample idx
        :param idx:
        :return:
        """
        if hasattr(self, '_cam_extrinsics_all'):
            # if not self.fix_cam:
            if 'openlane' in self.dataset_name:
                H_g2im = homograpthy_g2im_extrinsic(self._cam_extrinsics_all[idx], self._cam_intrinsics_all[idx])
                P_g2im = projection_g2im_extrinsic(self._cam_extrinsics_all[idx], self._cam_intrinsics_all[idx])
            else:
                H_g2im = homograpthy_g2im(self._label_cam_pitch_all[idx],
                                        self._label_cam_height_all[idx], self.K)
                P_g2im = projection_g2im(self._label_cam_pitch_all[idx],
                                        self._label_cam_height_all[idx], self.K)

            H_im2ipm = np.linalg.inv(np.matmul(self.H_crop, np.matmul(H_g2im, self.H_ipm2g)))
            return H_g2im, P_g2im, self.H_crop, H_im2ipm
            # else:
            #     return self.H_g2im, self.P_g2im, self.H_crop, self.H_im2ipm
        else:
            idx_json_file = self._label_list[idx]
            with open(idx_json_file, 'r') as file:
                file_lines = [line for line in file]
                info_dict = json.loads(file_lines[0])

                if not self.fix_cam:
                    if 'extrinsic' in info_dict:
                        cam_extrinsics = np.array(info_dict['extrinsic'])
                    else:
                        cam_pitch = 0.5/180*np.pi
                        cam_height = 1.5
                        cam_extrinsics = np.array([[np.cos(cam_pitch), 0, -np.sin(cam_pitch), 0],
                                                    [0, 1, 0, 0],
                                                    [np.sin(cam_pitch), 0,  np.cos(cam_pitch), cam_height],
                                                    [0, 0, 0, 1]], dtype=float)
                    # Re-calculate extrinsic matrix based on ground coordinate
                    R_vg = np.array([[0, 1, 0],
                                        [-1, 0, 0],
                                        [0, 0, 1]], dtype=float)
                    R_gc = np.array([[1, 0, 0],
                                        [0, 0, 1],
                                        [0, -1, 0]], dtype=float)
                    cam_extrinsics[:3, :3] = np.matmul(np.matmul(
                                                np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
                                                    R_vg), R_gc)
                    cam_extrinsics[0:2, 3] = 0.0
                    
                    # gt_cam_height = info_dict['cam_height']
                    gt_cam_height = cam_extrinsics[2, 3]
                    if 'cam_pitch' in info_dict:
                        gt_cam_pitch = info_dict['cam_pitch']
                    else:
                        gt_cam_pitch = 0

                    if 'intrinsic' in info_dict:
                        cam_intrinsics = info_dict['intrinsic']
                        cam_intrinsics = np.array(cam_intrinsics)
                    elif 'calibration' in info_dict:
                        cam_intrinsics = info_dict['calibration']
                        cam_intrinsics = np.array(cam_intrinsics)
                        cam_intrinsics = cam_intrinsics[:, :3]
                    else:
                        cam_intrinsics = self.K  

                _label_cam_height = gt_cam_height
                _label_cam_pitch = gt_cam_pitch

                return self.transform_mats_impl(cam_extrinsics, cam_intrinsics, _label_cam_pitch, _label_cam_height)

    def transform_mats_impl(self, cam_extrinsics, cam_intrinsics, cam_pitch, cam_height):
        H_g2im = homograpthy_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
        P_g2im = projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)

        H_im2ipm = np.linalg.inv(np.matmul(self.H_crop, np.matmul(H_g2im, self.H_ipm2g))).astype(np.float32)
        
        return H_g2im, P_g2im, self.H_crop, H_im2ipm

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane
