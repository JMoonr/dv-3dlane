import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from scipy.interpolate import UnivariateSpline

from experiments.gpu_utils import is_main_process


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(transformed_dataset, args, is_train=True):
    """
        create dataset from ground-truth
        return a batch sampler based ont the dataset
    """

    # transformed_dataset = LaneDataset(dataset_base_dir, json_file_path, args)
    sample_idx = range(transformed_dataset.n_samples)

    g = torch.Generator()
    g.manual_seed(0)

    # sample_idx = sample_idx[0:len(sample_idx)//args.batch_size*args.batch_size]
    discarded_sample_start = len(sample_idx) // args.batch_size * args.batch_size
    if is_main_process():
        print("Discarding images:")
        if hasattr(transformed_dataset, '_label_image_path'):
            print(transformed_dataset._label_image_path[discarded_sample_start: len(sample_idx)])
        else:
            print(len(sample_idx) - discarded_sample_start)
    sample_idx = sample_idx[0 : discarded_sample_start]
    
    if args.dist:
        if is_main_process():
            print('use distributed sampler')
        if 'standard' in args.dataset_name or 'rare_subset' in args.dataset_name or 'illus_chg' in args.dataset_name:
            data_sampler = torch.utils.data.distributed.DistributedSampler(transformed_dataset, shuffle=is_train, drop_last=not is_train)
            data_loader = DataLoader(transformed_dataset,
                                        batch_size=args.batch_size, 
                                        sampler=data_sampler,
                                        num_workers=args.nworkers, 
                                        pin_memory=True,
                                        persistent_workers=args.nworkers > 0,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        drop_last=True)
        else:
            data_sampler = torch.utils.data.distributed.DistributedSampler(transformed_dataset, shuffle=is_train, drop_last=not is_train)
            data_loader = DataLoader(transformed_dataset,
                                        batch_size=args.batch_size, 
                                        sampler=data_sampler,
                                        num_workers=args.nworkers, 
                                        pin_memory=True,
                                        persistent_workers=args.nworkers > 0,
                                        worker_init_fn=seed_worker,
                                        generator=g)
    else:
        if is_main_process():
            print("use default sampler")
        data_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idx)
        data_loader = DataLoader(transformed_dataset,
                                batch_size=args.batch_size, sampler=data_sampler,
                                num_workers=args.nworkers, pin_memory=True,
                                persistent_workers=args.nworkers > 0,
                                worker_init_fn=seed_worker,
                                generator=g)

    if args.dist:
        return data_loader, data_sampler
    return data_loader


def calculate_thickness(y, y_min=600, y_max=720):
    thickness_min = 5
    thickness_max = 10  # Maximum thickness value
    normalized_y = (y - y_min) / (y_max - y_min)  # Normalize y-coordinate
    thickness = thickness_min + normalized_y * (thickness_max - thickness_min)
    return int(thickness)


def increase_ys(y_2d, x_2d):
    inc_index = np.argsort(y_2d)
    y_2d = y_2d[inc_index]
    x_2d = x_2d[inc_index]
    return y_2d, x_2d


def decrease_ys(y_2d, x_2d):
    dec_index = np.argsort(-1 * y_2d)
    x_2d = x_2d[dec_index]
    y_2d = y_2d[dec_index]
    return y_2d, x_2d


def smooth_lanes(x_2d, y_2d):
    if len(y_2d) <= 1:
        return x_2d, y_2d
    y_2d, x_2d = increase_ys(y_2d, x_2d)
    # x_y_spline = scipy.interpolate.make_interp_spline(y_2d, x_2d)
    spl = UnivariateSpline(y_2d, x_2d, k=min(3, len(x_2d) - 1))
    x_2d = spl(y_2d)
    y_2d, x_2d = decrease_ys(y_2d, x_2d)
    return x_2d, y_2d