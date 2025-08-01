import argparse
import torch
import os
import numpy as np
# from mmcv.utils import Config, DictAction

from utils.config import Config, DictAction
from utils.utils import *
from experiments.ddp import *
from experiments.runner import *
from utils.misc import update_mod_for_workdir
from data.Load_Data_sequential import LaneDataset

from collections import defaultdict
from multiprocessing import Pool, cpu_count
import pickle


def build_dataset(args, split='training'):
    dataset = LaneDataset(
        args.dataset_dir, args.data_dir + f'{split}/', args, data_aug=False)
    return dataset


def parse_annos(args):
    json_file_path = os.path.join(
        args.dataset_cfg.data_label_dir + f'{split}/')

    split = 'train' if 'train' in json_file_path else 'val'
    global lidar_info_cache_path
    lidar_info_cache_path = f'.cache/infos_dict_{split}.pkl'

    if not os.path.exists(args.lidar_processed_dir):
        args.logger.error("waymo data path doesn't EXIST ! please check the '--lidar_processed_dir' input value: %s" % args.lidar_processed_dir)
        raise ValueError

    if os.path.isfile(lidar_info_cache_path):
        print(f' >>> Already existed: lidar info_dict from .cache: {lidar_info_cache_path}')
        exit()
    else:
        print(' >>> going to parse lidar info ...')
        train_annos_dir = f'{args.lidar_processed_dir}/train/annos'
        val_annos_dir = f'{args.lidar_processed_dir}/val/annos'

        train_info_tokens = os.listdir(train_annos_dir)
        train_info_tokens = [f'{train_annos_dir}/{_t}' for _t in train_info_tokens]

        val_info_tokens = os.listdir(val_annos_dir)
        val_info_tokens = [f'{val_annos_dir}/{_t}' for _t in val_info_tokens]

        all_info_tokens = train_info_tokens + val_info_tokens
        return all_info_tokens

def process_lidar_single(lidar_infos_dict, info_token):
    anno_pkl = info_token
    info_token = ops.basename(info_token)
    with open(anno_pkl, 'rb') as f:
        annos = pickle.load(f)
    frame_name = annos['frame_name'].split('_')
    timestamp = frame_name[-1]
    scene_name = annos['scene_name']
    frame_key = timestamp[:12]
    seg_name_key = 'segment-' + scene_name + '_with_camera_labels'
    # assert frame_key not in lidar_infos_dict[seg_name_key], 'frame_key already in lidar_infos_dict'
    # lidar_infos_dict[seg_name_key][frame_key] = dict(
    #     token=info_token,
    #     lidar_root=ops.dirname(anno_pkl).replace('annos', 'lidar')
    # )
    lidar_root = ops.dirname(anno_pkl).replace('annos', 'lidar')
    return seg_name_key, frame_key, info_token, lidar_root

def prepare_seq_lidar_data(args):

    def _read_extr(info):
        if 'extr' in info:
            return info['extr'].reshape(4, 4)
        if hasattr(args.args, 'lidar_root'):
            lidar_root = os.path.join(
                args.args.lidar_root,
                '/'.join(info['lidar_root'].split('/')[-2:]))
        else:
            lidar_root = info['lidar_root']
        token_file = ops.join(lidar_root[:-len('lidar')], 'annos', info['token'])
        with open(token_file, 'rb') as f:
            annos_info = pickle.load(f)
            extr = annos_info['veh_to_global']
            info['extr'] = extr
        return extr.reshape(4, 4)

    if args.sample_every_n_meter > 0:
        assert args.lidar_num_frames == 1, 'can not use both'
        # lidar_adj_info_cache_spec = f'.cache/infos_dict_adj_{args.sample_every_n_meter}m_{args.sample_total_length}m_{split}.pkl'
        lidar_adj_info_cache_spec = f'.cache/infos_dict_adj_{args.sample_every_n_meter}m_{args.sample_total_length}m_all.pkl'
        lidar_adj_info_cache = f'.cache/infos_dict_all_with_extr.pkl'

        if os.path.isfile(lidar_adj_info_cache_spec):
            print(f' >>> load lidar adj frames info_dict from {lidar_adj_info_cache_spec} ...')
            with open(lidar_adj_info_cache_spec, 'rb') as f:
                lidar_infos_dict = pickle.load(f)
        else:
            if os.path.isfile(lidar_adj_info_cache):
                print(f' >>> load lidar adj frames info_dict from {lidar_adj_info_cache} ...')
                with open(lidar_adj_info_cache, 'rb') as f:
                    lidar_infos_dict = pickle.load(f)
            else:
                print("Read all extr....")
                for scene_name, frames in tqdm(lidar_infos_dict.items()):
                    for frame_idx, info in frames.items():
                        _read_extr(info)
                with open(lidar_adj_info_cache, 'wb') as f:
                    pickle.dump(lidar_infos_dict, f)
            # import pdb;pdb.set_trace()
            print("Build adj frames....")
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

                        if delta_x > args.sample_every_n_meter:
                            adj_frames.append(next_frame)
                            cur_extr = next_extr

                            if np.linalg.norm((np.linalg.inv(next_extr) @ first_extr)[0:2, -1]) \
                                    > args.sample_total_length:
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
                for frame_offset in range(1, args.lidar_num_frames // args.lidar_frame_step):
                    if args.lidar_temporal_fuse_backward:
                        target_idx = cur_idx - frame_offset * args.args.frame_step
                    else:
                        target_idx = cur_idx + frame_offset * args.args.frame_step
                    if target_idx < 0 or target_idx >= len(sorted_all_frames):
                        adj_frames.append(False)
                    else:
                        adj_frames.append(sorted_all_frames[target_idx])
                v['adj_frames'] = adj_frames
    return lidar_infos_dict


def process_file(args):
    idx, idx_json_file, dataset_base_dir = args
    path_parts = idx_json_file.split('/')
    scene_name = path_parts[-2]
    timestamp_name = path_parts[-1].split('.')[0]
    timestamp_name = timestamp_name[:12]

    with open(idx_json_file, 'r') as file:
        file_lines = [line for line in file]
        info_dict = json.loads(file_lines[0])
        image_path = ops.join(dataset_base_dir, info_dict['file_path'])

        cam_extrinsics = np.array(info_dict['extrinsic'])
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

        assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    return scene_name, timestamp_name, dict(
        idx=idx, json_file=idx_json_file,
        image_path=image_path,
        extr=cam_extrinsics)

def process_scene(scene_name, processed_path, infos_dict):
    pkl_path = os.path.join(processed_path, scene_name, '%s.pkl' % scene_name)
    with open(pkl_path, 'rb') as f:
        seq_info = pickle.load(f)

    seq_info.sort(key=lambda x: int(x['metadata']['timestamp']))
    for frame_idx, info in enumerate(seq_info):
        timestamp = str(info['metadata']['timestamp'])[:12]
        if not timestamp in infos_dict[scene_name]:
            print(f"{scene_name}, {timestamp}, not exists")
            # infos_dict[scene_name][timestamp] = None
            continue
        infos_dict[scene_name][timestamp].update(
            dict(
                scene_name=scene_name,
                timestamp=timestamp,
                veh_to_global=info['veh_to_global']
            )
        )


def prepare_lidar_data(dataset):
    all_info_tokens = parse_annos(args)
    global lidar_info_cache_path
    assert not os.path.exists(lidar_info_cache_path):

    with Pool(16) as pool:
        for seg_name_key, frame_key, info_token, lidar_root in tqdm(pool.imap(process_lidar_single, all_info_tokens), total=len(all_info_tokens)):
            if scene_name not in infos_dict:
                infos_dict[scene_name] = {}
            assert frame_key not in infos_dict[scene_name]
            infos_dict[seg_name_key][frame_key] = dict(
                 token=info_token,
                 lidar_root=lidar_root)

    infos_dict_lidar_cache_path = os.path.join('.cache2/infos_dict_lidar_%s.pkl' % split_org)
    # train_processed = 158081
    scene_names = os.listdir(processed_path)
    with Pool(16) as pool:
        for scene_name in tqdm(scene_names):
            pool.apply_async(process_scene, args=(scene_name, processed_path, infos_dict))
        pool.close()
        pool.join()

    # import pdb;pdb.set_trace()
    with open(infos_dict_lidar_cache_path, "wb") as f:
        pickle.dump(infos_dict, f)
        

    for scene_name in infos_dict.keys():
        # pop none
        lidar_has_openlane_dont = []
        for k, v in infos_dict[scene_name].items():
            if v is None:
                lidar_has_openlane_dont.append(k)
        for k in lidar_has_openlane_dont:
            infos_dict[scene_name].pop(k)
    

def get_args():
    parser = argparse.ArgumentParser()
    # exp setting
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='overwrite config param.')
    return parser.parse_args()


def main():
    args = get_args()
    # define runner to begin training or evaluation
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    cfg.is_torch2 = True

    cfg = update_mod_for_workdir(cfg, args.config)
    cfg.merge_from_dict(vars(args))

    dataset = build_dataset(cfg)
    prepare_lidar_data(dataset)



    

if __name__ == '__main__':
    main()