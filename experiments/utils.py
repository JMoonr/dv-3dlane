import torch.distributed as dist
import torch

import os.path as ops
from data.Load_Data import LaneDataset
from data.data_utils import  get_loader

from experiments.gpu_utils import is_main_process


class OpenLaneSplit(object):
    Train = 'training'
    Val = 'validation'
    TrainV2 = 'train'
    ValV2 = 'val'
    UpDown = 'test/up_down_case'
    Curve = 'test/curve_case'
    ExWeather = 'test/extreme_weather_case'
    Intersect = 'test/intersection_case'
    MS = 'test/merge_split_case'
    Night = 'test/night_case'


def convert_to_cuda(data_dict):
    for k, v in data_dict.items():
        data_dict[k] = v.cuda()
    return data_dict


def get_dataset(args, split=OpenLaneSplit.Train, pipeline=None, logger=None, info_dict=None, is_train=True):
    if 'openlane' in args.dataset_name:
        split_name = eval(split)
    else:
        raise NotImplementedError(f'{args.dataset_name} not supported yet')

    if 'openlane' in args.dataset_name:
        data_class = LaneDataset
    else:
        raise NotImplementedError(f'{args.dataset_name} not supported yet')

    dataset = data_class(
                    args,
                    pipeline=pipeline,
                    logger=logger,
                    split=split_name,
                    info_dict=info_dict)

    loader, sampler = get_loader(dataset, args, is_train=is_train)

    return dataset, loader, sampler




if __name__ == '__main__':
    a = 'OpenLaneSplit.Train'
    print(eval(a))