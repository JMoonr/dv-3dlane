from mmdet3d.utils import build_from_cfg

from ..registry import READERS


def mm_build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_reader(cfg):
    return mm_build(cfg, READERS)
