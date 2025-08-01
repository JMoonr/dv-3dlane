import numpy as np
import torch
import cv2
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from mmdet3d.datasets.pipelines import (
            PointSample, PointShuffle, Compose, PointsRangeFilter)

from models.scatter_utils import scatter_mean


class LidarEncodingManner:
    ORIGINAL_PT_FEAT = 0
    ADD_PT_IMAGE_RGB = 1
    ADD_PT_SEG_LABEL = 2


def build_points_pipelines(piplines_cfg):
    piplines = []
    for pipline in piplines_cfg:
        op_type = eval(pipline.pop('type'))
        piplines.append(op_type(**pipline))
    return Compose(piplines)


def get_homo_coords(normal_coords, coords_first=False, with_return_T=False):
    """
    normal_coords: N x 2 / N x 3 -> coords_first = False
    normal_coords: 2 x N / 3 x N -> coords_first = True
    """
    if coords_first:
        # 2 x N / 3 x N
        homo_coords = np.concatenate([
            normal_coords, 
            np.ones((1, normal_coords.shape[0]))], 
            axis=0)
    else:
        # N x 2 / N x 3
        homo_coords = np.concatenate([
            normal_coords, 
            np.ones((normal_coords.shape[0], 1))],
            axis=-1)
    if with_return_T:
        return homo_coords.T
    

def lidar2cam(points, cam_extr):
    """
    points : Nx(3+c)
    cam_extr : 4x4
    """
    xyz = points[:, :3]
    xyz = np.concatenate(
        [xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)],
        axis=-1)
    xyz = (np.linalg.inv(cam_extr) @ xyz.T).T
    return np.concatenate([xyz[:, :3], points[:, 3:]], axis=-1)


def cam2img(points, cam_intr):
    """
    points : Nx(3+c)
    cam_intr : 3x3
    """
    xyz = points[:, :3]
    xyz_new = (cam_intr @ xyz.T).T

    zeros = np.isclose(xyz[:, 2:3], 0.0)  # Find zero elements
    denominator = np.where(zeros, 1e-8, xyz[:, 2:3])  # Replace zero elements with 1.0
    xy = xyz_new[:, :2] / denominator

    # xy = xyz_new[:, :2] / xyz[:, 2:3]
    return np.concatenate([xy, points[:, 3:]], axis=-1), xyz[:, 2]


def filter_fov(points, depth, h, w):
    xy = points[:, :2]
    mask = (xy[:, 0] >= 0) * \
           (xy[:, 0] < w) * \
           (xy[:, 1] >=0) * \
           (xy[:, 1] < h) * \
           (depth >= 0)
    return points[mask], mask


def get_lidar_fov_feat(
    enc_feat_mtd, 
    lidar_fov_feat_org,
    image,
    img_pt):
    """
    enc_feat_method   : way to attach feat to the org lidar feat.
    lidar_fov_feat_org: original waymo lidar feat, inten & elong.
    image             : Image.
    img_pt            : projected lidar pts to img with cam params.
    """
    # TODO using RGB color from image / using seg labels -> lidar_points
    if enc_feat_mtd == LidarEncodingManner.ADD_PT_IMAGE_RGB:
        rgb = np.array(image)[
            np.clip(img_pt[:, 1].astype(np.int), 0, image.height - 1),
            np.clip(img_pt[:, 0].astype(np.int), 0, image.width - 1)]
        lidar_fov_feat_org = np.concatenate([lidar_fov_feat_org, rgb], axis=1)
    elif enc_feat_mtd == LidarEncodingManner.ADD_PT_SEG_LABEL:
        raise NotImplementedError('TODO, using seg label for lidar')
    else:
        lidar_fov_feat_org = lidar_fov_feat_org
    return lidar_fov_feat_org


def get_points_rgb(points_uv: np.ndarray,
                #    depth: np.ndarray, 
                   image: Image):
    img_x = points_uv[:, 0]
    img_y = points_uv[:, 1]
    if isinstance(image, Tensor):
        H, W = image.shape[1:]
        assert image.shape[0] == 3
        rgb_pt = image
    else:
        raise ValueError

    img_x_norm = (img_x / W) * 2 - 1
    img_y_norm = (img_y / H) * 2 - 1

    coords_pt = torch.from_numpy(
        np.stack([img_x_norm, img_y_norm], axis=-1)
    ).float()

    points_rgb = F.grid_sample(
        rgb_pt.unsqueeze(0),
        coords_pt[None, None, ...]
    )
    points_rgb = points_rgb[0, :, 0] # .permute(1, 0).contiguous()
    return points_rgb




if __name__ == '__main__':
    # N x 3
    pts = np.random.rand(10, 3)
    homo_pts = get_homo_coords(pts)
    assert homo_pts.shape[0] == 10 and homo_pts.shape[1] == 4
