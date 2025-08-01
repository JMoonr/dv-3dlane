# ==============================================================================
# Binaries and/or source for the following packages or projects are presented under one or more of the following open
# source licenses:
# eval_3D_lane.py       The OpenLane Dataset Authors        Apache License, Version 2.0
#
# Contact simachonghao@pjlab.org.cn if you have any issue
# 
# See:
# https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection/blob/master/tools/eval_3D_lane.py
#
# Copyright (c) 2022 The OpenLane Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Description: This code is to evaluate 3D lane detection. The optimal matching between ground-truth set and predicted 
    set of lanes are sought via solving a min cost flow.

Evaluation metrics includes:
    F-scores
    x error close (0 - 40 m)
    x error far (0 - 100 m)
    z error close (0 - 40 m)
    z error far (0 - 100 m)
"""

from copy import deepcopy
import numpy as np
from utils.utils import *
from utils.MinCostFlow import SolveMinCostFlow

# # try:
# from openlanev2.dataset import Collection
# from openlanev2.evaluation import evaluate as openlanev2_evaluate
# from openlanev2.preprocessing import check_results
# from openlanev2.visualization.utils import COLOR_DICT
# # except Exception as e:
# #     openlanev2_evaluate = None


class LaneEval(object):
    def __init__(self, args):        
        self.dataset_name = args.dataset_name
        self.dataset_dir = args.data_image_dir

        self.x_min = args.top_view_region[0, 0]
        self.x_max = args.top_view_region[1, 0]
        self.y_min = args.top_view_region[2, 1]
        self.y_max = args.top_view_region[0, 1]
        self.y_samples = np.linspace(self.y_min, self.y_max, num=100, endpoint=False)
        # self.dist_th = 1.5
        self.dist_th = args.get('dist_th', 0.5)
        self.ratio_th = 0.75
        self.close_range = 40
        print("self.dist_th ->>>>>>>>>>>>>>>>>>", self.dist_th)

    def bench(self, pred_lanes, pred_category, gt_lanes, gt_visibility, gt_category, raw_file, gt_cam_height, gt_cam_pitch, vis, P_g2im=None):
        """
            Matching predicted lanes and ground-truth lanes in their IPM projection, ignoring z attributes.
            x error, y_error, and z error are all considered, although the matching does not rely on z
            The input of prediction and ground-truth lanes are in ground coordinate, x-right, y-forward, z-up
            The fundamental assumption is: 1. there are no two points from different lanes with identical x, y
                                              but different z's
                                           2. there are no two points from a single lane having identical x, y
                                              but different z's
            If the interest area is within the current drivable road, the above assumptions are almost always valid.

        :param pred_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param gt_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param raw_file: file path rooted in dataset folder
        :param gt_cam_height: camera height given in ground-truth data
        :param gt_cam_pitch: camera pitch given in ground-truth data
        :return:
        """
        
        # change this properly
        close_range_idx = np.where(self.y_samples > self.close_range)[0][0]

        r_lane, p_lane, c_lane = 0., 0., 0.
        x_error_close = []
        x_error_far = []
        z_error_close = []
        z_error_far = []

        # only keep the visible portion
        gt_lanes = [prune_3d_lane_by_visibility(np.array(gt_lane), np.array(gt_visibility[k])) for k, gt_lane in
                    enumerate(gt_lanes)]
        if 'openlane' in self.dataset_name:
            gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        # # only consider those pred lanes overlapping with sampling range
        pred_category = [pred_category[k] for k, lane in enumerate(pred_lanes)
                        if np.array(lane)[0, 1] < self.y_samples[-1] and np.array(lane)[-1, 1] > self.y_samples[0]]
        pred_lanes = [lane for lane in pred_lanes if np.array(lane)[0, 1] < self.y_samples[-1] and np.array(lane)[-1, 1] > self.y_samples[0]]

        pred_lanes = [prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in pred_lanes]

        pred_category = [pred_category[k] for k, lane in enumerate(pred_lanes) if np.array(lane).shape[0] > 1]
        pred_lanes = [lane for lane in pred_lanes if np.array(lane).shape[0] > 1]

        # only consider those gt lanes overlapping with sampling range
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes)
                        if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]

        gt_lanes = [prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in gt_lanes]

        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        cnt_gt = len(gt_lanes)
        cnt_pred = len(pred_lanes)

        gt_visibility_mat = np.zeros((cnt_gt, 100))
        pred_visibility_mat = np.zeros((cnt_pred, 100))

        # resample gt and pred at y_samples
        for i in range(cnt_gt):
            min_y = np.min(np.array(gt_lanes[i])[:, 1])
            max_y = np.max(np.array(gt_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(gt_lanes[i]), self.y_samples, out_vis=True)
            gt_lanes[i] = np.vstack([x_values, z_values]).T
            gt_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, np.logical_and(x_values <= self.x_max,
                                                     np.logical_and(self.y_samples >= min_y, self.y_samples <= max_y)))
            gt_visibility_mat[i, :] = np.logical_and(gt_visibility_mat[i, :], visibility_vec)

        for i in range(cnt_pred):
            # # ATTENTION: ensure y mono increase before interpolation: but it can reduce size
            # pred_lanes[i] = make_lane_y_mono_inc(np.array(pred_lanes[i]))
            # pred_lane = prune_3d_lane_by_range(np.array(pred_lanes[i]), self.x_min, self.x_max)
            min_y = np.min(np.array(pred_lanes[i])[:, 1])
            max_y = np.max(np.array(pred_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(pred_lanes[i]), self.y_samples, out_vis=True)
            pred_lanes[i] = np.vstack([x_values, z_values]).T
            pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, np.logical_and(x_values <= self.x_max,
                                                       np.logical_and(self.y_samples >= min_y, self.y_samples <= max_y)))
            pred_visibility_mat[i, :] = np.logical_and(pred_visibility_mat[i, :], visibility_vec)
            # pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min, x_values <= self.x_max)

        # at least two-points for both gt and pred
        gt_lanes = [gt_lanes[k] for k in range(cnt_gt) if np.sum(gt_visibility_mat[k, :]) > 1]
        gt_category = [gt_category[k] for k in range(cnt_gt) if np.sum(gt_visibility_mat[k, :]) > 1]
        gt_visibility_mat = gt_visibility_mat[np.sum(gt_visibility_mat, axis=-1) > 1, :]
        cnt_gt = len(gt_lanes)

        pred_lanes = [pred_lanes[k] for k in range(cnt_pred) if np.sum(pred_visibility_mat[k, :]) > 1]
        pred_category = [pred_category[k] for k in range(cnt_pred) if np.sum(pred_visibility_mat[k, :]) > 1]
        pred_visibility_mat = pred_visibility_mat[np.sum(pred_visibility_mat, axis=-1) > 1, :]
        cnt_pred = len(pred_lanes)

        adj_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
        cost_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
        cost_mat.fill(1000)
        num_match_mat = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_close.fill(1000.)
        x_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_far.fill(1000.)
        z_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=float)
        z_dist_mat_close.fill(1000.)
        z_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=float)
        z_dist_mat_far.fill(1000.)

        # compute curve to curve distance
        for i in range(cnt_gt):
            for j in range(cnt_pred):
                x_dist = np.abs(gt_lanes[i][:, 0] - pred_lanes[j][:, 0])
                z_dist = np.abs(gt_lanes[i][:, 1] - pred_lanes[j][:, 1])

                # apply visibility to penalize different partial matching accordingly
                both_visible_indices = np.logical_and(gt_visibility_mat[i, :] >= 0.5, pred_visibility_mat[j, :] >= 0.5)
                both_invisible_indices = np.logical_and(gt_visibility_mat[i, :] < 0.5, pred_visibility_mat[j, :] < 0.5)
                other_indices = np.logical_not(np.logical_or(both_visible_indices, both_invisible_indices))
                
                euclidean_dist = np.sqrt(x_dist ** 2 + z_dist ** 2)
                euclidean_dist[both_invisible_indices] = 0
                euclidean_dist[other_indices] = self.dist_th

                # if np.average(euclidean_dist) < 2*self.dist_th: # don't prune here to encourage finding perfect match
                num_match_mat[i, j] = np.sum(euclidean_dist < self.dist_th) - np.sum(both_invisible_indices)
                adj_mat[i, j] = 1
                # ATTENTION: use the sum as int type to meet the requirements of min cost flow optimization (int type)
                # using num_match_mat as cost does not work?
                cost_ = np.sum(euclidean_dist)
                if cost_<1 and cost_>0:
                    cost_ = 1
                else:
                    cost_ = (cost_).astype(int)
                cost_mat[i, j] = cost_

                # use the both visible portion to calculate distance error
                if np.sum(both_visible_indices[:close_range_idx]) > 0:
                    x_dist_mat_close[i, j] = np.sum(
                        x_dist[:close_range_idx] * both_visible_indices[:close_range_idx]) / np.sum(
                        both_visible_indices[:close_range_idx])
                    z_dist_mat_close[i, j] = np.sum(
                        z_dist[:close_range_idx] * both_visible_indices[:close_range_idx]) / np.sum(
                        both_visible_indices[:close_range_idx])
                else:
                    x_dist_mat_close[i, j] = -1
                    z_dist_mat_close[i, j] = -1
                    

                if np.sum(both_visible_indices[close_range_idx:]) > 0:
                    x_dist_mat_far[i, j] = np.sum(
                        x_dist[close_range_idx:] * both_visible_indices[close_range_idx:]) / np.sum(
                        both_visible_indices[close_range_idx:])
                    z_dist_mat_far[i, j] = np.sum(
                        z_dist[close_range_idx:] * both_visible_indices[close_range_idx:]) / np.sum(
                        both_visible_indices[close_range_idx:])
                else:
                    x_dist_mat_far[i, j] = -1
                    z_dist_mat_far[i, j] = -1

        # solve bipartite matching vis min cost flow solver
        match_results = SolveMinCostFlow(adj_mat, cost_mat)
        match_results = np.array(match_results)

        # only a match with avg cost < self.dist_th is consider valid one
        match_gt_ids = []
        match_pred_ids = []
        match_num = 0
        if match_results.shape[0] > 0:
            for i in range(len(match_results)):
                if match_results[i, 2] < self.dist_th * self.y_samples.shape[0]:
                    match_num += 1
                    gt_i = match_results[i, 0]
                    pred_i = match_results[i, 1]
                    # consider match when the matched points is above a ratio
                    if num_match_mat[gt_i, pred_i] / np.sum(gt_visibility_mat[gt_i, :]) >= self.ratio_th:
                        r_lane += 1
                        match_gt_ids.append(gt_i)
                    if num_match_mat[gt_i, pred_i] / np.sum(pred_visibility_mat[pred_i, :]) >= self.ratio_th:
                        p_lane += 1
                        match_pred_ids.append(pred_i)
                    if pred_category != []:
                        if pred_category[pred_i] == gt_category[gt_i] or (pred_category[pred_i]==20 and gt_category[gt_i]==21):
                            c_lane += 1    # category matched num
                    x_error_close.append(x_dist_mat_close[gt_i, pred_i])
                    x_error_far.append(x_dist_mat_far[gt_i, pred_i])
                    z_error_close.append(z_dist_mat_close[gt_i, pred_i])
                    z_error_far.append(z_dist_mat_far[gt_i, pred_i])
        # # Visulization to be added
        # if vis:
        #     pass 
        return r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, x_error_close, x_error_far, z_error_close, z_error_far

    def bench_one_submit(self, pred_dir, gt_dir, test_txt, prob_th=0.5, vis=False):
        pred_lines = open(test_txt).readlines()
        gt_lines = pred_lines

        json_pred = []
        json_gt = []

        print("Loading pred json ...")
        for pred_file_path in pred_lines:
            pred_lines = pred_dir + pred_file_path.strip('\n').replace('jpg','json')

            with open(pred_lines,'r') as fp:
                json_pred.append(json.load(fp))

        print("Loading gt json ...")
        for gt_file_path in gt_lines:
            gt_lines = gt_dir + gt_file_path.strip('\n').replace('jpg','json')

            with open(gt_lines,'r') as fp:
                json_gt.append(json.load(fp))
                
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')

        gts = {l['file_path']: l for l in json_gt}

        laneline_stats = []
        laneline_x_error_close = []
        laneline_x_error_far = []
        laneline_z_error_close = []
        laneline_z_error_far = []
        for i, pred in enumerate(json_pred):
            if i % 1000 == 0 or i == len(json_pred)-1:
                print('eval:{}/{}'.format(i+1,len(json_pred)))
            if 'file_path' not in pred or 'lane_lines' not in pred:
                raise Exception('file_path or lane_lines not in some predictions.')
            raw_file = pred['file_path']

            pred_lanelines = pred['lane_lines']
            pred_lanes = [np.array(lane['xyz']) for i, lane in enumerate(pred_lanelines)]
            pred_category = [int(lane['category']) for i, lane in enumerate(pred_lanelines)]
            
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]

            # evaluate lanelines
            cam_extrinsics = np.array(gt['extrinsic'])
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
            gt_cam_height = cam_extrinsics[2, 3]
            gt_cam_pitch = 0

            cam_extrinsics[0:2, 3] = 0.0
            # cam_extrinsics[2, 3] = gt_cam_height

            cam_intrinsics = gt['intrinsic']
            cam_intrinsics = np.array(cam_intrinsics)

            try:
                gt_lanes_packed = gt['lane_lines']
            except:
                print("error 'lane_lines' in gt: ", gt['file_path'])

            gt_lanes, gt_visibility, gt_category = [], [], []
            for j, gt_lane_packed in enumerate(gt_lanes_packed):
                # A GT lane can be either 2D or 3D
                # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                lane = np.array(gt_lane_packed['xyz'])
                lane_visibility = np.array(gt_lane_packed['visibility'])

                lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
                cam_representation = np.linalg.inv(
                                        np.array([[0, 0, 1, 0],
                                                  [-1, 0, 0, 0],
                                                  [0, -1, 0, 0],
                                                  [0, 0, 0, 1]], dtype=float))
                lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))
                lane = lane[0:3, :].T

                gt_lanes.append(lane)
                gt_visibility.append(lane_visibility)
                gt_category.append(gt_lane_packed['category'])
            
            P_g2im = projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)

            # N to N matching of lanelines
            r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, \
            x_error_close, x_error_far, \
            z_error_close, z_error_far = self.bench(pred_lanes,
                                                    pred_category, 
                                                    gt_lanes,
                                                    gt_visibility,
                                                    gt_category,
                                                    raw_file,
                                                    gt_cam_height,
                                                    gt_cam_pitch,
                                                    vis,
                                                    P_g2im)
            laneline_stats.append(np.array([r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num]))
            # consider x_error z_error only for the matched lanes
            # if r_lane > 0 and p_lane > 0:
            laneline_x_error_close.extend(x_error_close)
            laneline_x_error_far.extend(x_error_far)
            laneline_z_error_close.extend(z_error_close)
            laneline_z_error_far.extend(z_error_far)


        output_stats = []
        laneline_stats = np.array(laneline_stats)
        laneline_x_error_close = np.array(laneline_x_error_close)
        laneline_x_error_far = np.array(laneline_x_error_far)
        laneline_z_error_close = np.array(laneline_z_error_close)
        laneline_z_error_far = np.array(laneline_z_error_far)

        if np.sum(laneline_stats[:, 3])!= 0:
            R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 3]))
        else:
            R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 3]) + 1e-6)   # recall = TP / (TP+FN)
        if np.sum(laneline_stats[:, 4]) != 0:
            P_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 4]))
        else:
            P_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 4]) + 1e-6)   # precision = TP / (TP+FP)
        if np.sum(laneline_stats[:, 5]) != 0:
            C_lane = np.sum(laneline_stats[:, 2]) / (np.sum(laneline_stats[:, 5]))
        else:
            C_lane = np.sum(laneline_stats[:, 2]) / (np.sum(laneline_stats[:, 5]) + 1e-6)   # category_accuracy
        if R_lane + P_lane != 0:
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane)
        else:
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)
        x_error_close_avg = np.average(laneline_x_error_close[laneline_x_error_close > -1 + 1e-6])
        x_error_far_avg = np.average(laneline_x_error_far[laneline_x_error_far > -1 + 1e-6])
        z_error_close_avg = np.average(laneline_z_error_close[laneline_z_error_close > -1 + 1e-6])
        z_error_far_avg = np.average(laneline_z_error_far[laneline_z_error_far > -1 + 1e-6])

        output_stats.append(F_lane)
        output_stats.append(R_lane)
        output_stats.append(P_lane)
        output_stats.append(C_lane)
        output_stats.append(x_error_close_avg)
        output_stats.append(x_error_far_avg)
        output_stats.append(z_error_close_avg)
        output_stats.append(z_error_far_avg)
        output_stats.append(np.sum(laneline_stats[:, 0]))   # 8
        output_stats.append(np.sum(laneline_stats[:, 1]))   # 9
        output_stats.append(np.sum(laneline_stats[:, 2]))   # 10
        output_stats.append(np.sum(laneline_stats[:, 3]))   # 11
        output_stats.append(np.sum(laneline_stats[:, 4]))   # 12
        output_stats.append(np.sum(laneline_stats[:, 5]))   # 13

        return output_stats

    # compare predicted set and ground-truth set using a fixed lane probability threshold
    def bench_one_submit_openlane_DDP(self, pred_lines_sub, gt_lines_sub, model_name, prob_th=0.5, vis=False):
        json_gt = gt_lines_sub
        json_pred = pred_lines_sub

        gts = {l['file_path']: l for l in json_gt}

        laneline_stats = []
        laneline_x_error_close = []
        laneline_x_error_far = []
        laneline_z_error_close = []
        laneline_z_error_far = []

        gt_num_all, pred_num_all = 0, 0
        for i, pred in enumerate(json_pred):
            # if i % 1000 == 0 or i == len(json_pred)-1:
            #     print('eval:{}/{}'.format(i+1,len(json_pred)))
            if 'file_path' not in pred or 'lane_lines' not in pred:
                raise Exception('file_path or lane_lines not in some predictions.')
            raw_file = pred['file_path']

            pred_lanes = pred['pred_laneLines']
            pred_lanes_prob = pred['pred_laneLines_prob']
            if model_name == "GenLaneNet":
                pred_lanes = [pred_lanes[ii] for ii in range(len(pred_lanes_prob)) if
                              pred_lanes_prob[ii] > prob_th]
                pred_category = np.zeros(len(pred_lanes_prob))
            # Note: non-lane class is already filtered out in compute_3d_lanes_all_category()
            # import pdb; pdb.set_trace()
            else:
                pred_lanes = [pred_lanes[ii] for ii in range(len(pred_lanes_prob)) if max(pred_lanes_prob[ii]) > prob_th]
                pred_lanes_prob = [prob for k, prob in enumerate(pred_lanes_prob) if max(prob) > prob_th]
                if pred_lanes_prob:
                    pred_category = np.argmax(pred_lanes_prob, 1)
                else:
                    pred_category = []

            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]

            # evaluate lanelines
            cam_extrinsics = np.array(gt['extrinsic'])
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
            gt_cam_height = cam_extrinsics[2, 3]
            gt_cam_pitch = 0

            cam_extrinsics[0:2, 3] = 0.0
            # cam_extrinsics[2, 3] = gt_cam_height

            cam_intrinsics = gt['intrinsic']
            cam_intrinsics = np.array(cam_intrinsics)

            try:
                gt_lanes_packed = gt['lane_lines']
            except:
                print("error 'lane_lines' in gt: ", gt['file_path'])

            gt_lanes, gt_visibility, gt_category = [], [], []
            for j, gt_lane_packed in enumerate(gt_lanes_packed):
                # A GT lane can be either 2D or 3D
                # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                lane = np.array(gt_lane_packed['xyz'])
                lane_visibility = np.array(gt_lane_packed['visibility'])

                lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
                cam_representation = np.linalg.inv(
                                        np.array([[0, 0, 1, 0],
                                                  [-1, 0, 0, 0],
                                                  [0, -1, 0, 0],
                                                  [0, 0, 0, 1]], dtype=float))
                lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))
                lane = lane[0:3, :].T

                gt_lanes.append(lane)
                gt_visibility.append(lane_visibility)
                gt_category.append(gt_lane_packed['category'])
            
            P_g2im = projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)

            # N to N matching of lanelines
            gt_num_all += len(gt_lanes)
            pred_num_all += len(pred_lanes)
            r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, \
            x_error_close, x_error_far, \
            z_error_close, z_error_far = self.bench(pred_lanes,
                                                    pred_category, 
                                                    gt_lanes,
                                                    gt_visibility,
                                                    gt_category,
                                                    raw_file,
                                                    gt_cam_height,
                                                    gt_cam_pitch,
                                                    vis,
                                                    P_g2im)
            laneline_stats.append(np.array([r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num]))
            # consider x_error z_error only for the matched lanes
            # if r_lane > 0 and p_lane > 0:
            laneline_x_error_close.extend(x_error_close)
            laneline_x_error_far.extend(x_error_far)
            laneline_z_error_close.extend(z_error_close)
            laneline_z_error_far.extend(z_error_far)

        output_stats = []
        laneline_stats = np.array(laneline_stats)

        laneline_x_error_close = np.array(laneline_x_error_close)
        laneline_x_error_far = np.array(laneline_x_error_far)
        laneline_z_error_close = np.array(laneline_z_error_close)
        laneline_z_error_far = np.array(laneline_z_error_far)

        print("match num:"+(str(np.sum(laneline_stats[:,5]))))
        print("cnt_gt_all:"+str(gt_num_all))
        print("cnt_pred_all:"+str(pred_num_all))
        print("cnt_gt_matched:"+str(np.sum(laneline_stats[:,3])))
        print("cnt_pred_matched:"+str(np.sum(laneline_stats[:,4])))

        
        if np.sum(laneline_stats[:, 3])!= 0:
            R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 3]))
        else:
            R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 3]) + 1e-6)   # recall = TP / (TP+FN)
        if np.sum(laneline_stats[:, 4]) != 0:
            P_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 4]))
        else:
            P_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 4]) + 1e-6)   # precision = TP / (TP+FP)
        if np.sum(laneline_stats[:, 5]) != 0:
            C_lane = np.sum(laneline_stats[:, 2]) / (np.sum(laneline_stats[:, 5]))
        else:
            C_lane = np.sum(laneline_stats[:, 2]) / (np.sum(laneline_stats[:, 5]) + 1e-6)   # category_accuracy
        if (R_lane + P_lane) != 0:
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane)
        else:
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)
        x_error_close_avg = np.average(laneline_x_error_close[laneline_x_error_close > -1 + 1e-6])
        x_error_far_avg = np.average(laneline_x_error_far[laneline_x_error_far > -1 + 1e-6])
        z_error_close_avg = np.average(laneline_z_error_close[laneline_z_error_close > -1 + 1e-6])
        z_error_far_avg = np.average(laneline_z_error_far[laneline_z_error_far > -1 + 1e-6])

        output_stats.append(F_lane)
        output_stats.append(R_lane)
        output_stats.append(P_lane)
        output_stats.append(C_lane)
        output_stats.append(x_error_close_avg)
        output_stats.append(x_error_far_avg)
        output_stats.append(z_error_close_avg)
        output_stats.append(z_error_far_avg)
        output_stats.append(np.sum(laneline_stats[:, 0]))   # 8
        output_stats.append(np.sum(laneline_stats[:, 1]))   # 9
        output_stats.append(np.sum(laneline_stats[:, 2]))   # 10
        output_stats.append(np.sum(laneline_stats[:, 3]))   # 11
        output_stats.append(np.sum(laneline_stats[:, 4]))   # 12
        output_stats.append(np.sum(laneline_stats[:, 5]))   # 13

        return output_stats


# class LaneV2Eval(object):
#     def __init__(self, args):
#         assert openlanev2_evaluate is not None, 'Install openlane v2'


#     def format_preds(self, results):

#         predictions = {
#             'method': 'dummy',
#             'authors': ['dummy'],
#             'e-mail': 'dummy',
#             'institution / company': 'dummy',
#             # 'country / region': None,
#             'results': {},
#         }
#         for index, result in enumerate(results):
#             prediction = {                
#                 'lane_centerline': [],
#                 'traffic_element': [],
#                 'topology_lclc': None,
#                 'topology_lcte': None,
#             }

#             # lc

#             pred_lc = result['pred_lc']
#             lanes, confidences = pred_lc[0], pred_lc[1][:, 0]

#             lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)

#             def comb(n, k):
#                 return factorial(n) // (factorial(k) * factorial(n - k))
#             n_points = 11
#             n_control = lanes.shape[1]
#             A = np.zeros((n_points, n_control))
#             t = np.arange(n_points) / (n_points - 1)
#             for i in range(n_points):
#                 for j in range(n_control):
#                     A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
#             bezier_A = torch.tensor(A, dtype=torch.float32)
#             lanes = torch.tensor(lanes, dtype=torch.float32)
#             lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
#             lanes = lanes.numpy()

#             for i, (lane, confidence) in enumerate(zip(lanes, confidences)):
#                 prediction['lane_centerline'].append({
#                     'id': i + 1000,
#                     'points': lane.astype(np.float32),
#                     'confidence': confidence,
#                 })

#             # te

#             pred_te = result['pred_te']
#             for i, (bbox, confidence) in enumerate(zip(*pred_te)):
#                 prediction['traffic_element'].append({
#                     'id': i + 2000,
#                     'attribute': bbox[-1],
#                     'points': bbox[:-1].reshape(2, 2).astype(np.float32),
#                     'confidence': confidence,
#                 })

#             # topology

#             prediction['topology_lclc'] = result['pred_topology_lclc']
#             prediction['topology_lcte'] = result['pred_topology_lcte']

#             #

#             predictions['results'][self.data_infos[index]] = {
#                 'predictions': prediction,
#             }

#         return predictions


#     def evaluate(self, 
#                  results, 
#                  logger=None,
#                  dump=None,
#                  dump_dir=None,
#                  visualization=False, 
#                  visualization_dir=None,
#                  visualization_num=None,
#                  **kwargs):
        
#         if logger:
#             logger.info(f'Start formating...')
#         pred_dict = self.format_preds(results)

#         if dump:
#             assert dump_dir is not None
#             assert check_results(pred_dict), "Please fill the missing keys."
#             output_path = os.path.join(dump_dir, 'result.pkl')
#             mmcv.dump(pred_dict, output_path)

#         if visualization:
#             assert visualization_dir is not None
#             self.visualize(pred_dict, visualization_dir, visualization_num, **kwargs)
        
#         if logger:
#             logger.info(f'Start evaluatation...')
#         metric_results = {}
#         for key, val in openlanev2_evaluate(ground_truth=self.ann_file, predictions=pred_dict).items():
#             for k, v in val.items():
#                 metric_results[k if k != 'score' else key] = v
#         return metric_results

#     # def format_preds(self, results):