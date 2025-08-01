import os.path as osp
from pathlib import Path
import configparser
import torch.distributed as dist


def is_main_process() -> bool:
    return get_rank() == 0


# for Config usage.

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def define_params(cfg):
    # work around error caused by importing lib in cfg.
    # if cfg.top_view_region is not None:
    #     return cfg
    # top_view_region = np.array([
    # [-10, 105.4], [10, 105.4], [-10, 3], [10, 3]])
    # cfg.position_range = [
    #     top_view_region[0][0] - cfg.enlarge_length,
    #     top_view_region[2][1],# - enlarge_length,
    #     -5,
    #     top_view_region[1][0] + cfg.enlarge_length,
    #     top_view_region[0][1],# + enlarge_length,
    #     5.]
    # cfg.top_view_region = top_view_region
    # cfg.anchor_y_steps = np.linspace(3, 103, 20)
    # cfg.num_y_steps = len(cfg.anchor_y_steps)

    # decoder = cfg.transformer.decoder
    # deform_attn = decoder.transformerlayers.attn_cfgs[-1]

    # decoder.anchor_y_steps = cfg.anchor_y_steps
    # deform_attn.anchor_y_steps = cfg.anchor_y_steps
    # work around error in deform_attn, duplicated as one in mmcv. 
    if deform_attn.type == 'MSDeformableAttention3D':
        deform_attn.type = 'LATR' + deform_attn.type
    assert deform_attn.type == 'LATRMSDeformableAttention3D'
    
    return cfg


def update_mod_for_workdir(cfg, config_path):
    config_path_obj = Path(config_path)
    work_dir = '/'.join([config_path_obj.parents[0].stem, config_path_obj.stem])
    mod_is_correct = cfg.mod == work_dir
    
    if mod_is_correct:
        return cfg

    cfg.mod = work_dir

    with open(config_path, 'r') as file:
        config_lines = file.readlines()

    new_assignment_line = f"mod = '{work_dir}'\n"
    
    tgt_idx = -1
    for idx, line in enumerate(config_lines):
        if line.strip().startswith('mod'):
            tgt_idx = idx
            break
    
    if tgt_idx == -1:
        config_lines.append(new_assignment_line)
    else:
        config_lines[tgt_idx] = new_assignment_line

    dist.barrier()
    
    if is_main_process():
        # Write the updated config back to the file
        with open(config_path, 'w') as file:
            file.writelines(config_lines)
    
    return cfg
