import numpy as np
import mmcv
import torch
import torch.nn.functional as F
from PIL import Image

import cv2
import numpy as np
import random

try:
    from mmdet3d.datasets.builder import PIPELINES
except ImportError as e:
    from mmcv.datasets.builder import PIPELINES

"""
    Data Augmentation: 
        idea 1: (currently in use)
            when initializing dataset, all labels will be prepared in 3D which do not need to be changed in image augmenting
            Image data augmentation would change the spatial transform matrix integrated in the network, provide 
            the transformation matrix related to random cropping, scaling and rotation
        idea 2:
            Introduce random sampling of cam_h, cam_pitch and their associated transformed image
            img2 = [R2[:, 0:2], T2] [R1[:, 0:2], T1]^-1 img1
            output augmented hcam, pitch, and img2 and untouched 3D anchor label value, Before forward pass, update spatial
            transform in network. However, However, image rotation is not considered, additional cropping is still needed
"""

@PIPELINES.register_module()
class RandomRot(object):
    def __init__(self, aug_rot_range=(-np.pi/18, np.pi/18)):
        assert isinstance(aug_rot_range, (tuple, list))
        self.aug_rot_range = aug_rot_range

    def __call__(self, img_dict):
        img = img_dict['img']
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        # assume img in PIL image format
        rot = random.uniform(*self.aug_rot_range)
        # rot = random.uniform(-10, 10)
        center_x = img.width / 2
        center_y = img.height / 2
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
        img_rot = np.array(img)
        img_rot = cv2.warpAffine(img_rot, rot_mat, (img.width, img.height), flags=cv2.INTER_LINEAR)
        # img_rot = img.rotate(rot)
        # rot = rot / 180 * np.pi
        rot_mat = np.vstack([rot_mat, [0, 0, 1]])

        img_dict['img'] = img_rot
        img_dict['rot_mat'] = rot_mat
        return img_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(aug_rot_range={self.aug_rot_range})'


@PIPELINES.register_module()
class NormIntensity(object):
    def __init__(self, intens_dim=3, use_tanh=True):
        self.intens_dim = intens_dim
        if use_tanh:
            self.norm = np.tanh
        else:
            self.norm = np.log
    
    def __call__(self, input_dict):
        input_dict['points'][:, self.intens_dim] = self.norm(input_dict['points'][:, self.intens_dim])
        return input_dict
    

@PIPELINES.register_module()
class GlobalRotScaleTransImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
        self,
        rotz_range=[-0.3925, 0.3925],
        rotx_range=[-0.2, 0.2],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        flip_x_prob=0.,
        reverse_angle=False,
        training=True,
    ):

        self.rotz_range = rotz_range
        self.rotx_range = rotx_range

        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.flip_x_prob = flip_x_prob
        self.training = training

    def __call__(self, input_dict):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        # random rotate
        lidar2img = input_dict['lidar2img']
        gt_lanes  = input_dict['gt_lanes']
        points    = input_dict['points']

        rotz_angle = np.random.uniform(*self.rotz_range)
        lidar2img, rotz_m = self.rotate_bev_along_z(
            lidar2img, rotz_angle)

        rotx_angle = np.random.uniform(*self.rotx_range)
        lidar2img, rotx_m = self.rotate_bev_along_x(
            lidar2img, rotx_angle)

        if self.flip_x_prob and np.random.uniform() > self.flip_x_prob:
            import pdb; pdb.set_trace()

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        lidar2img, trans_m = self.scale_xyz(lidar2img, scale_ratio)
        # results["gt_bboxes_3d"].scale(scale_ratio)

        for idx, lane in enumerate(gt_lanes):
            new_lane = \
                F.pad(torch.tensor(lane).float(),
                      (0, 1), value=1) @ rotz_m.T @ rotx_m.T @ trans_m.T
            gt_lanes[idx] = new_lane[:, :3].cpu().numpy()

        if points is not None:
            xyz_homo = torch.cat([
                points[:, :3], torch.ones((points.shape[0], 1))
            ], dim=-1)
            xyz_new = xyz_homo @ rotz_m.T @ rotx_m.T @ trans_m.T
            points[:, :3] = xyz_new[:, :3]

        input_dict['lidar2img'] = lidar2img
        input_dict['gt_lanes']  = gt_lanes
        input_dict['points']    = points
        return input_dict

    def rotate_bev_along_z(self, lidar2img, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([
            [rot_cos, -rot_sin, 0, 0],
            [rot_sin, rot_cos, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        rot_mat_inv = torch.inverse(rot_mat)
        lidar2img = (torch.tensor(lidar2img).float() \
                @ rot_mat_inv).numpy()
        # results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ rot_mat_inv).numpy()
        return lidar2img, rot_mat

    def rotate_bev_along_x(self, lidar2img, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([
            [1, 0, 0, 0],
            [0, rot_cos, -rot_sin, 0],
            [0, rot_sin, rot_cos, 0],
            [0, 0, 0, 1]
        ])
        rot_mat_inv = torch.inverse(rot_mat)
        lidar2img = (torch.tensor(lidar2img).float() \
                @ rot_mat_inv).numpy()
        # results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ rot_mat_inv).numpy()
        return lidar2img, rot_mat


    def scale_xyz(self, lidar2img, scale_ratio):
        rot_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        rot_mat_inv = torch.inverse(rot_mat)
        # for view in range(num_view):
        lidar2img = (torch.tensor(lidar2img).float() \
                @ rot_mat_inv).numpy()
        #     results["extrinsics"][view] = (torch.tensor(rot_mat_inv.T @ results["extrinsics"][view]).float()).numpy()
        return lidar2img, rot_mat


@PIPELINES.register_module()
class GlobalRotScaleHomoImage(object):
    def __init__(
        self,
        rotz_range=[-0.3925, 0.3925],
        rotx_range=[-0.2, 0.2],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
        fix_cam_param_ratio=1.0,
        prob=1.0,
        prob_rotx=1.0,
        prob_rotz=1.0,
        prob_scale=1.0,
        prob_trans_x=1.0,
        prob_trans_y=1.0,
        prob_trans_z=1.0,
        trans_x_type='normal',
        trans_y_type='normal',
        trans_z_type='normal',
        filter_points=None,
    ):
        self.rotz_range = rotz_range
        self.rotx_range = rotx_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.reverse_angle = reverse_angle
        self.training = training
        self.fix_cam_param_ratio = fix_cam_param_ratio
        self.prob = prob
        self.prob_rotx = prob_rotx
        self.prob_rotz = prob_rotz
        self.prob_trans_x = prob_trans_x
        self.prob_trans_y = prob_trans_y
        self.prob_trans_z = prob_trans_z
        self.trans_x_type = trans_x_type
        self.trans_y_type = trans_y_type
        self.trans_z_type = trans_z_type
        self.filter_points = filter_points

    def project(self, xyz, lidar2img, H, W):
        assert xyz.ndim == 2
        xyz_homo = np.concatenate(
            [xyz[:, :3], np.ones((xyz.shape[0], 1))], axis=1)
        points_img = xyz_homo @ lidar2img.T
        points_xy = points_img[..., :2] / np.clip(
            points_img[..., 2:3], 1e-5, points_img.max())
        valid = (points_xy[..., 0] >= 0) * (points_xy[..., 1] >= 0) * \
                (points_xy[..., 0] < W) * (points_xy[..., 1] < H) * \
                (points_img[..., 2] > 0)
        return points_xy, valid

    def get_random_translate(self, param, t):
        if t == 'normal':
            return np.random.normal(param)
        elif t == 'uniform':
            return np.random.uniform(*param)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if np.random.uniform() > self.prob:
            return results

        lidar2img_ori = lidar2img = results['lidar2img']
        gt_lanes = results['gt_lanes']
        H = results['H']
        W = results['W']
        points = results.get('points', None)

        fix_cam_param = np.random.uniform() < self.fix_cam_param_ratio

        if len(gt_lanes) < 2:
            return results

        dx = dy = dz = 0.
        if np.random.uniform() < self.prob_trans_x:
            dx = self.get_random_translate(self.translation_std[0], self.trans_x_type)
        if np.random.uniform() < self.prob_trans_y:
            dy = self.get_random_translate(self.translation_std[1], self.trans_y_type)
        if np.random.uniform() < self.prob_trans_z:
            dz = self.get_random_translate(self.translation_std[2], self.trans_z_type)
        lidar2img, trans_t = self.trans_xyz(lidar2img, dx, dy, dz, False)

        # random rotate
        rotz_range = self.rotz_range
        if rotz_range[1] > rotz_range[0]:
            rotz_angle = np.random.uniform(*self.rotz_range)
            lidar2img, rotz_m = self.rotate_bev_along_z(
                lidar2img, rotz_angle, False)
        else:
            rotz_m = torch.eye(4)

        rotx_range = self.rotx_range
        if rotx_range[1] > rotx_range[0]:
            rotx_angle = np.random.uniform(*self.rotx_range)
            lidar2img, rotx_m = self.rotate_bev_along_x(
                lidar2img, rotx_angle, False)
        else:
            rotx_m = torch.eye(4)

        scale_ratio_range = self.scale_ratio_range
        if scale_ratio_range[1] > scale_ratio_range[0]:
            scale_ratio = np.random.uniform(*self.scale_ratio_range)
            lidar2img, trans_m = self.scale_xyz(
                lidar2img, scale_ratio, False)
        else:
            trans_m = torch.eye(4)

        new_gt_lanes = [None for _ in gt_lanes]
        for idx, lane in enumerate(gt_lanes):
            new_lane = \
                F.pad(torch.tensor(lane).float(),
                    (0, 1), value=1) @ trans_t.T @ rotz_m.T @ rotx_m.T @ trans_m.T
            new_gt_lanes[idx] = new_lane[:, :3].cpu().numpy()

        if points is not None:
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points).float()
            if self.filter_points is not None:
                num_points = points.shape[0]
                minx = points[:, 0].min()
                maxx = points[:, 0].max()
                miny = points[:, 1].min()
                maxy = points[:, 1].max()
            xyz_homo = torch.cat([
                points[:, :3], torch.ones((points.shape[0], 1))
            ], dim=-1)
            xyz_new = xyz_homo @ trans_t.T @ rotz_m.T @ rotx_m.T @ trans_m.T
            points[:, :3] = xyz_new[:, :3]

            if self.filter_points is not None:
                assert False
                mask = torch.logical_and(
                    torch.logical_and(
                    torch.logical_and(
                    xyz_new[..., 0] >= minx,
                    xyz_new[..., 0] <= maxx),
                    xyz_new[..., 1] >= miny),
                    xyz_new[..., 1] <= maxy)
                points = points[mask]
                resample_idx = np.random.choice(
                    np.arange(points.shape[0]),
                    size=num_points, replace=True)
                points = points[resample_idx]

            results['points'] = points
        results['gt_lanes'] = new_gt_lanes

        if fix_cam_param:
            all_points = np.concatenate(gt_lanes, axis=0)
            new_all_points = np.concatenate(new_gt_lanes, axis=0)

            old_pts_uv, old_mask = self.project(all_points, lidar2img_ori, H, W)
            new_pts_uv, new_mask = self.project(new_all_points, lidar2img_ori, H, W)

            in_mask = (old_mask > 0) * (new_mask > 0)
            if in_mask.sum() < 10:
                results['lidar2img'] = lidar2img
                return results

            choosen4pts_idx = np.random.choice(
                np.arange(all_points.shape[0])[in_mask > 0],
                size=4, replace=False)
            old4pts = old_pts_uv[choosen4pts_idx]
            new4pts = new_pts_uv[choosen4pts_idx]
            pT = cv2.getPerspectiveTransform(
                old4pts.astype(np.float32), new4pts.astype(np.float32))

            if (pT[0][0] < 0 and pT[0][1]) < 0 \
                    or (pT[1][0] < 0 and pT[1][1] < 0) \
                    or np.abs(pT[:, -1]).max() > 200:
                results['lidar2img'] = lidar2img
                return results

            results['pT'] = pT # perspective_trans
        else:
            results['lidar2img'] = lidar2img
        return results

    def rotate_bev_along_z(self, lidar2img, angle, fix_cam_param):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, -rot_sin, 0, 0], [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        if not fix_cam_param:
            rot_mat_inv = torch.inverse(rot_mat)
            lidar2img = (torch.tensor(lidar2img).float() \
                    @ rot_mat_inv).numpy()
        return lidar2img, rot_mat

    def rotate_bev_along_x(self, lidar2img, angle, fix_cam_param):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([
            [1, 0, 0, 0],
            [0, rot_cos, -rot_sin, 0],
            [0, rot_sin, rot_cos, 0],
            [0, 0, 0, 1]
        ])

        if not fix_cam_param:
            rot_mat_inv = torch.inverse(rot_mat)
            lidar2img = (torch.tensor(lidar2img).float() \
                    @ rot_mat_inv).numpy()
        return lidar2img, rot_mat

    def scale_xyz(self, lidar2img, scale_ratio, fix_cam_param):
        rot_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        if not fix_cam_param:
            rot_mat_inv = torch.inverse(rot_mat)
            lidar2img = (torch.tensor(lidar2img).float() \
                    @ rot_mat_inv).numpy()
        return lidar2img, rot_mat

    def trans_xyz(self, lidar2img, dx, dy, dz, fix_cam_param):
        rot_mat = torch.tensor(
            [
                [1, 0, 0, dx],
                [0, 1, 0, dy],
                [0, 0, 1, dz],
                [0, 0, 0, 1],
            ]
        )

        if not fix_cam_param:
            rot_mat_inv = torch.inverse(rot_mat)
            lidar2img = (torch.tensor(lidar2img).float() \
                    @ rot_mat_inv).numpy()
        return lidar2img, rot_mat


@PIPELINES.register_module(force=True)
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 rgb2bgr=False):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.rgb2bgr = rgb2bgr

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        if not isinstance(imgs, list):
            imgs = [imgs]

        new_imgs = []
        for img in imgs:
            if img.dtype is not np.float32:
                img = img.astype(np.float32)
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if np.random.randint(2):
                delta = np.random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            if self.rgb2bgr:
                img = mmcv.rgb2bgr(img)
            img = mmcv.bgr2hsv(img)

            # random saturation
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if np.random.randint(2):
                img[..., 0] += np.random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)
            if self.rgb2bgr:
                img = mmcv.bgr2rgb(img)

            # random contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if np.random.randint(2):
                img = img[..., np.random.permutation(3)]
            new_imgs.append(img)
        if not isinstance(results['img'], list):
            new_imgs = new_imgs[0]

        results['img'] = new_imgs
        return results

