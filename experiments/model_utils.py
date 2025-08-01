import os
import glob
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from models.dv3dlane import DV3DLane
from experiments.gpu_utils import is_main_process
from utils.utils import define_init_weights


def get_model(args):
    model = DV3DLane(args)
    return model


def syn_bn(args, model, logger):
    if args.sync_bn:
        if is_main_process():
            logger.info("Convert model with Sync BatchNorm")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

def to_cuda(args, model):
    if not args.no_cuda:
        device = torch.device("cuda", args.local_rank)
        model = model.to(device)
        return model


def load_eval_ckpt(args, model, logger):
    eval_ckpt = args.eval_ckpt
    if eval_ckpt:
        best_file_name = eval_ckpt
    else:
        best_file_name = glob.glob(os.path.join(args.save_path, 'model_best*'))
        if len(best_file_name) > 0:
            best_file_name = best_file_name[0]
        else:
            best_file_name = ''
    if os.path.isfile(best_file_name):
        checkpoint = torch.load(best_file_name)
        if is_main_process():
            logger.info("=> loading checkpoint '{}'".format(best_file_name))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logger.info("=> no checkpoint found at '{}'".format(best_file_name))
    return model


def ddp_model(args, model):
    find_unused_parameters = getattr(args, 'find_unused_parameters', False)
    print('dist: %s' % args.distributed)
    if args.distributed:
        model = DDP(model, 
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    find_unused_parameters=find_unused_parameters)
        return model


def log_model_params(args, model, logger):
    if is_main_process():
        logger.info(40*"="+"\nArgs:{}\n".format(args)+40*"=")
        logger.info("Init model: '{}'".format(args.mod))
        logger.info("Number of parameters in model {} is {:.3f}M".format(args.mod, sum(tensor.numel() for tensor in model.parameters())/1e6))


def log_lr(optimizer, logger):
    if is_main_process():
        lr = optimizer.param_groups[0]['lr']
        logger.info('lr is set to {}'.format(lr))


def log_epoch(epoch, logger, epoch_txt='Start train set'):
    if is_main_process():
        logger.info(f"\n => {epoch_txt} for EPOCH {epoch+1}")


def log_train_info(args, 
                   logger,
                   iter_i, 
                   epoch, 
                   loader_len, 
                   batch_time, 
                   data_time, 
                   loss, 
                   loss_info,
                   neper=None):
    if (iter_i + 1) % args.print_freq == 0 and is_main_process():
        logger.info('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss:.8f} {loss_info}'.format(
                epoch+1, iter_i+1, loader_len, 
                batch_time=batch_time, data_time=data_time,
                loss=loss.item(), loss_info=loss_info))
        if neper is not None:
            neper["train/batch/loss"].append(loss.item())
