import numpy as np
import cv2


def near_one_pt(pts):
    """
    pts: Nx2,
    return: whether the max and min of pts are int(max-min) == 0
    """
    return (int(pts[:, 0].max()) - int(pts[:, 0].min())) == 0 and (int(pts[:, 1].max()) - int(pts[:, 1].min())) == 0


def get_vis_mask(y_steps, lane3d, 
                 near_dist=40, 
                 pre_toldist=2,
                 tol_dist=5):
    y_min = np.min(lane3d[:, 1])
    y_max = np.max(lane3d[:, 1])
    
    # if y_min <= near_dist:
    #     y_min = max(0, min(np.min(y_steps), y_min))
    # else:
        # y_min -= tol_dist
    y_vis = np.logical_and(
        y_steps > y_min - pre_toldist,
        y_steps < y_max + tol_dist
    )
    return y_vis


def draw_2d_lanes(imgs: list, 
                  lanes_on_imgs: list, 
                  colors: list, 
                  thickness: list):
    assert len(lanes_on_imgs) == len(colors)
    if isinstance(thickness, int):
        thickness = [thickness for _ in range(len(lanes_on_imgs))]
    elif len(thickness) == 1:
        thickness = thickness * len(imgs)
    else:
        raise ValueError('thickness to draw seg requires int / equal length list as imgs.')
    
    for idx, lanes in enumerate(lanes_on_imgs):
        if lanes.shape[0] == 1 or near_one_pt(lanes):
            imgs[idx] = cv2.circle(imgs[idx], 
                                    tuple(map(np.int32, lanes)),
                                    thickness[idx], colors[idx], -1)
        else:
            imgs[idx] = cv2.polylines(
                imgs[idx],
                [np.int32(lanes).reshape((-1, 1, 2))],
                isClosed=False,
                color=colors[idx],
                thickness=thickness[idx]
            )
    return imgs

           