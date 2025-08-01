import numpy as np
import torch
from scipy.signal import savgol_filter
# from data.ransac import ransac_filter
from scipy.signal import convolve2d
from shapely.geometry import LineString


def fix_pts_interpolate(lane, n_points=11):
    try:
        ls = LineString(lane)
    except Exception as e:
        print(lane.shape)
        import pdb;pdb.set_trace()
    distances = np.linspace(0, ls.length, n_points)
    lane = np.array([ls.interpolate(distance).coords[0] for distance in distances])
    return lane


def bilateral_filter_1d(points, spatial_sigma, intensity_sigma):
    result_points = np.zeros_like(points)

    diff = points[:, 0] - points[:, 0][:, np.newaxis] # 0 for y
    spatial_kernel = np.exp(-0.5 * (diff**2) / spatial_sigma**2)

    intensity_diff = points[:, 1] - points[:, 1][:, np.newaxis] # 1 for z
    intensity_kernel = np.exp(-0.5 * (intensity_diff**2) / intensity_sigma**2)

    # Compute weights
    weights = spatial_kernel * intensity_kernel

    # Normalize weights
    weights /= np.sum(weights, axis=0)

    # Apply weights to each dimension
    result_points[:, 0] = np.sum(weights * points[:, 1][:, np.newaxis], axis=0)

    return result_points[:, 0]
