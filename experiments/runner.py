import torch
import torch.optim
import torch.nn as nn
import numpy as np
import glob
import time
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import traceback
import shutil
import torch.distributed as dist
from mmcv.runner.optimizer import build_optimizer
from mmcv.runner.hooks import HOOKS

from data.Load_Data import *
from experiments.gpu_utils import is_main_process
from experiments.model_utils import get_model, syn_bn, to_cuda, ddp_model, load_eval_ckpt
from experiments.model_utils import log_model_params, log_epoch, log_lr, log_train_info # , write_tb
from experiments.utils import get_dataset
from experiments.utils import convert_to_cuda
from experiments.manager import CkptResStateManager, EvalManager, log_eval_stats
from utils.utils import *

from .ddp import *


class Runner:
    def __init__(self, args, neper=None):
        self.args = args
        self.neper = None
        save_id = args.mod
        args.save_json_path = args.save_path
        args.save_path = os.path.join(args.save_path, save_id)
        
        self.logger = create_logger(args)
        # Check GPU availability
        if is_main_process():
            if not args.no_cuda and not torch.cuda.is_available():
                raise Exception("No gpu available for usage")
            if int(os.getenv('WORLD_SIZE', 1)) >= 1:
                self.logger.info("Let's use %s" % os.environ['WORLD_SIZE'] + "GPUs!")
                torch.cuda.empty_cache()

        if is_main_process():
            mkdir_if_missing(args.save_path)
            shutil.copy(args.config, os.path.join(args.save_path, os.path.basename(args.config)))

        # Get Dataset
        if is_main_process():
            self.logger.info("Loading Dataset ...")

        if not self.args.evaluate:
            self.train_dataset, self.train_loader, self.train_sampler = get_dataset(
                args, 
                split=args.dataset_cfg.train_split, 
                pipeline=args.dataset_cfg.train_pipeline,
                logger=self.logger, is_train=True)
        self.valid_dataset, self.valid_loader, self.valid_sampler = get_dataset(
            args, 
            split=args.dataset_cfg.val_split,
            pipeline=args.dataset_cfg.val_pipeline,
            logger=self.logger,
            info_dict=self.train_dataset.lidar_infos_dict if hasattr(self, 'train_dataset') else None,
            is_train=False)
        self.eval_manager = EvalManager(args)
        
        if is_main_process():
            tensorboard_path = os.path.join(args.save_path, 'Tensorboard/')
            mkdir_if_missing(tensorboard_path)
            self.writer = SummaryWriter(tensorboard_path)
            self.logger.info("Init Done!")

        self.ckpt_res_state_tracker = CkptResStateManager(logger=self.logger)
        print('init done.')

    def train(self):
        args = self.args

        # Get Dataset
        train_dataset = self.train_dataset
        train_loader = self.train_loader
        train_sampler = self.train_sampler
        model, optimizer, scheduler, ckpt_res_state = self._get_model_ddp()
        log_model_params(args, model, self.logger)

        # Start training and validation for nepochs
        print('start training')
        for epoch in range(args.start_epoch, args.nepochs):
            log_epoch(epoch, self.logger)
            log_lr(optimizer, self.logger)

            if args.distributed:
                train_sampler.set_epoch(epoch)

            if epoch > args.seg_start_epoch:
                args.loss_seg_weight = 10.0

            # Define container objects to keep track of multiple losses/metrics
            batch_time = AverageMeter()
            data_time = AverageMeter()          # compute FPS
            losses = AverageMeter()
            losses_3d_vis = AverageMeter()
            losses_3d_prob = AverageMeter()
            losses_3d_reg = AverageMeter()
            losses_2d_vis = AverageMeter()
            losses_2d_cls = AverageMeter()
            losses_2d_reg = AverageMeter()

            # Specify operation modules
            model.train()
            # compute timing
            end = time.time()
            # Start training loop
            train_pbar = tqdm(total=len(train_loader), ncols=60)
            for i, extra_dict in enumerate(train_loader):
                train_pbar.update(1)
                data_time.update(time.time() - end)
                if not args.no_cuda:
                    json_files = extra_dict.pop('idx_json_file')
                    extra_dict = convert_to_cuda(extra_dict)
                    image = extra_dict['image']
                image = image.contiguous().float()
                extra_dict['epoch'] = epoch

                # Run model
                optimizer.zero_grad()
                output = model(image=image, extra_dict=extra_dict, is_training=True)
                loss = 0.0
                loss_info = ''
                for k, v in output.items():
                    if 'loss' in k:
                        loss = loss + v
                        loss_info = loss_info + '| %s:%.4f ' % (k, v.item() if isinstance(v, torch.Tensor) else v)
                        if isinstance(v, torch.Tensor):
                            v = v.item()
                        if is_main_process():
                            self.writer.add_scalar(k, v, epoch*len(train_loader) + i)
                            if self.neper is not None:
                                self.neper['train/batch/%s' % k] = v

                train_pbar.set_postfix(loss=loss.item())
                loss.backward()

                # Clip gradients (usefull for instabilities or mistakes in ground truth)
                if args.clip_grad_norm != 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                # update params
                optimizer.step()

                if args.lr_policy == 'cosine_warmup':
                    scheduler.step(epoch + i / len(train_loader))
                # Time trainig iteration
                batch_time.update(time.time() - end)
                end = time.time()

                # Print info
                log_train_info(args, self.logger, iter_i=i, epoch=epoch, 
                               loader_len=len(self.train_loader), 
                               batch_time=batch_time, data_time=data_time,
                               loss=loss, loss_info=loss_info,
                               neper=self.neper
                                )
            train_pbar.close()

            # Adjust learning rate
            if args.lr_policy != 'cosine_warmup':
                scheduler.step()

            try:
                self.to_val_and_save_ckpt(model, 
                                          optimizer, 
                                          scheduler, 
                                          training_loss=loss, 
                                          epoch=epoch)
                
            except Exception as e:
                self.save_cur_ckpt_detailed(model, 
                                            optimizer,
                                            scheduler,
                                            loss, 
                                            epoch=epoch, 
                                            with_eval_res=False)
                self.logger.info("[WARNING] Eval with following error, skip this and continue training")
                self.logger.info("===================================================================")
                self.logger.info(''.join(traceback.extract_stack().format()))
                self.logger.info(traceback.format_exc())
                self.logger.info("===================================================================")

            dist.barrier()
            torch.cuda.empty_cache()

        # at the end of training
        if not args.no_tb and is_main_process():
            self.writer.close()

    def record_loss(self, eval_stats, epoch):
        if len(eval_stats) == 0:
            return
        eval_stats_name = ['F1', 'Recall', 'Precision', 'Acc', 'Error/x_near', 'Error/x_far', 'Error/z_near', 'Error/z_far']
        if is_main_process():
            for name, val in zip(eval_stats_name, eval_stats[:8]):
                self.writer.add_scalar(name, val, epoch)
                if self.neper is not None:
                    self.neper['val/%s' % name] = val

    def validate(self, model, model2=None, epoch=0, vis=False):
        args = self.args
        loader = self.valid_loader
        dataset = self.valid_dataset

        losses = AverageMeter()
        losses_3d_vis = AverageMeter()
        losses_3d_prob = AverageMeter()
        losses_3d_reg = AverageMeter()
        losses_2d_vis = AverageMeter()
        losses_2d_cls = AverageMeter()
        losses_2d_reg = AverageMeter()

        pred_lines_sub = []
        gt_lines_sub = []

        # Evaluate model
        model.eval()

        # Start validation loop
        with torch.no_grad():
            val_pbar = tqdm(total=len(loader), ncols=50)
            for i, extra_dict in enumerate(loader):
                val_pbar.update(1)

                if not args.no_cuda:
                    json_files = extra_dict.pop('idx_json_file')
                    extra_dict = convert_to_cuda(extra_dict)
                    image = extra_dict['image']
                image = image.contiguous().float()

                # if args.model_name == "LanePETR":
                output = model(image=image, extra_dict=extra_dict, is_training=False)
                all_line_preds = output['all_line_preds'] # in ground coordinate system
                all_cls_scores = output['all_cls_scores']

                all_line_preds = all_line_preds[-1]
                all_cls_scores = all_cls_scores[-1]
                num_el = all_cls_scores.shape[0]

                # Print info
                if (i + 1) % args.print_freq == 0 and is_main_process():
                    self.logger.info('Test: [{0}/{1}]'.format(i+1, len(loader)))

                # Write results
                for j in range(num_el):
                    json_file = json_files[j]
                    with open(json_file, 'r') as file:
                        file_lines = [line for line in file]
                        json_line = json.loads(file_lines[0])
                    json_line['json_file'] = json_file
                    if 'once' in args.dataset_name:
                        if 'training' in json_file:
                            img_path = json_file.replace('training', 'data').replace('.json', '.jpg')
                        elif 'validation' in json_file:
                            img_path = json_file.replace('validation', 'data').replace('.json', '.jpg')
                        elif 'test' in json_file:
                            img_path = json_file.replace('test', 'data').replace('.json', '.jpg')
                        json_line["file_path"] = img_path
                    gt_lines_sub.append(copy.deepcopy(json_line))

                    # pred in ground
                    lane_pred = all_line_preds[j].cpu().numpy()
                    cls_pred = torch.argmax(all_cls_scores[j], dim=-1).cpu().numpy()

                    pos_lanes = lane_pred[cls_pred > 0]
                    scores_pred = torch.softmax(all_cls_scores[j][cls_pred > 0], dim=-1).cpu().numpy()
                    if pos_lanes.shape[0]:
                        lanelines_pred = [] # [[] for _ in range(pos_lanes.shape[0])]
                        lanelines_prob = []
                        xs = pos_lanes[:, 0:args.num_y_steps]
                        ys = np.tile(args.anchor_y_steps.copy()[None, :], (xs.shape[0], 1))
                        zs = pos_lanes[:, args.num_y_steps:2*args.num_y_steps]
                        vis = pos_lanes[:, 2*args.num_y_steps:]

                        for tmp_idx in range(pos_lanes.shape[0]):
                            cur_vis = vis[tmp_idx] > 0
                            cur_xs = xs[tmp_idx][cur_vis]
                            cur_ys = ys[tmp_idx][cur_vis]
                            cur_zs = zs[tmp_idx][cur_vis]

                            if cur_vis.sum() < 2:
                                continue

                            lanelines_pred.append([])
                            for tmp_inner_idx in range(cur_xs.shape[0]):
                                lanelines_pred[-1].append(
                                    list(map(float, [cur_xs[tmp_inner_idx],
                                     cur_ys[tmp_inner_idx],
                                     cur_zs[tmp_inner_idx]])))
                            lanelines_prob.append(scores_pred[tmp_idx].tolist())
                    else:
                        lanelines_pred = []
                        lanelines_prob = []

                    json_line["pred_laneLines"] = lanelines_pred
                    json_line["pred_laneLines_prob"] = lanelines_prob
                    pred_lines_sub.append(copy.deepcopy(json_line))

                    if args.get('save_pred', False):
                        out_json_path = os.path.join(
                            args.save_path, 'preds',
                            *json_file.split('/')[-2:]
                        )
                        out_json_dir = os.path.dirname(out_json_path)
                        os.makedirs(out_json_dir, exist_ok=True)
                        with open(out_json_path, 'w') as f:
                            json.dump(json_line, f)

            val_pbar.close()

            # TODO
            eval_stats = self.eval_manager.eval_pred(
                                pred_lines_sub, 
                                gt_lines_sub, 
                                args, 
                                logger=self.logger)
        return eval_stats
            

    def eval(self):
        args = self.args
        
        #TODO8
        model = get_model(args)
        model = syn_bn(args, model, self.logger)
        model = to_cuda(args, model)
        
        model = load_eval_ckpt(args, model, self.logger)
        dist.barrier()
        # DDP setting
        model = ddp_model(args, model)
        eval_stats = self.validate(model, None, vis=True)

    def _get_model_ddp(self):
        args = self.args
        # Define network
        
        model = get_model(args)
        model = syn_bn(args, model, self.logger)
        model = to_cuda(args, model)
        """
            first load param to model, then model = DDP(model)
        """
        ckpt_res_state = dict(
            best_epoch = 0,
            lowest_loss = np.inf,
            best_f1_epoch = 0,
            best_val_f1 = -1e-5,
        )
        optim_saved_state, schedule_saved_state = None, None

        # resume model
        if args.resume_from:
            model, ckpt_res_state, optim_saved_state, schedule_saved_state = self.resume_model(args, model, args.resume_from)
        else:
            args.resume = first_run(args.save_path)
            if args.resume:
                model, ckpt_res_state, optim_saved_state, schedule_saved_state = self.resume_model(args, model)

        dist.barrier()
        # DDP setting
        model = ddp_model(args, model)
        optimizer = build_optimizer(
            model,
            args.optimizer_cfg
        )

        scheduler = define_scheduler(optimizer, args, dataset_size=len(self.train_loader))

        # resume optimizer and scheduler
        if optim_saved_state is not None:
            self.logger.info("proc_id-{} load optim state".format(args.proc_id))
            optimizer.load_state_dict(optim_saved_state)
        if schedule_saved_state is not None:
            self.logger.info("proc_id-{} load scheduler state".format(args.proc_id))
            scheduler.load_state_dict(schedule_saved_state)

        return model, optimizer, scheduler, ckpt_res_state

    def resume_model(self, args, model, path=None):
        if path is None:
            path = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(int(args.resume)))
        if not os.path.isfile(path):
            # try best
            path = os.path.join(args.save_path, 'model_best_epoch_{}.pth.tar'.format(int(args.resume)))
        
        ckpt_res_state = dict(
            best_epoch = 0,
            lowest_loss = np.inf,
            best_f1_epoch = 0,
            best_val_f1 = -1e-5,

        )
        if os.path.isfile(path):
            self.logger.info("=> loading checkpoint from {}".format(path))
            checkpoint = torch.load(path, map_location='cpu')
            if is_main_process():
                model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            ckpt_res_state['best_epoch'] = checkpoint['best_epoch']
            ckpt_res_state['lowest_loss'] = checkpoint['lowest_loss']
            ckpt_res_state['best_f1_epoch'] = checkpoint['best_f1_epoch']
            ckpt_res_state['best_val_f1'] = checkpoint['best_val_f1']
            optim_saved_state = checkpoint['optimizer']
            schedule_saved_state = checkpoint['scheduler']
            self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if is_main_process():
                self.logger.info("=> Warning: no checkpoint found at '{}'".format(path))
            optim_saved_state = None
            schedule_saved_state = None
        return model, ckpt_res_state, optim_saved_state, schedule_saved_state

    def save_checkpoint(self, state, to_copy, epoch):
        def extract_epoch(filename):
            base = os.path.basename(filename)
            epoch_str = base.split('_')[-1].split('.')[0]
            return int(epoch_str)

        if is_main_process():
            save_path = self.args.save_path
            self.logger.info('Saving checkpoint to {}'.format(save_path))
            filepath = os.path.join(save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(epoch))
            torch.save(state, filepath)
            if to_copy:
                if epoch > 2:
                    lst = glob.glob(os.path.join(save_path, 'model_best_epoch_*.pth.tar'))
                    lst = sorted(lst, key=extract_epoch)
                    if len(lst) != 0:
                        os.remove(lst[0])
                shutil.copyfile(filepath, os.path.join(save_path, 'model_best_epoch_{}.pth.tar'.format(epoch)))
                self.logger.info("Best model copied")
                os.remove(filepath)
                self.logger.info('only save best for the same model.')

    def to_val_and_save_ckpt(self, model, optimizer, scheduler, training_loss, epoch):
        args = self.args
        if self.args.eval_freq > 0 and (epoch + 1) % self.args.eval_freq == 0:
            eval_stats = self.validate(model, epoch, vis=False)
            with_eval_res = True
            if not self.args.no_tb:
                self.record_loss(eval_stats, epoch)
        else:
            eval_stats = None
            with_eval_res = False
        # if (epoch + 1) % self.args.save_freq == 0:
        self.save_cur_ckpt_detailed(model,
                                    optimizer,
                                    scheduler,
                                    training_loss, 
                                    epoch=epoch, 
                                    with_eval_res=with_eval_res,
                                    eval_stats=eval_stats)
        if is_main_process():
            # todo keep n ckpts
            ckpts = os.path.join(args.save_path, 'checkpoint_model_epoch_*.pth.tar')
            ckpts = glob.glob(ckpts)
            ckpts = sorted(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            for ckpt in ckpts[:-args.keep_n_ckpts]:
                os.remove(ckpt)
    
    def save_cur_ckpt_detailed(self,
                               model,
                               optimizer,
                               scheduler,
                               loss,
                               epoch,
                               with_eval_res=True,
                               eval_stats=None):
        args = self.args
        to_copy=False 
        state_dict = {
                'arch': self.args.mod,
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
        # training loss
        cur_loss = loss.item() # loss_list[0].avg
        if is_main_process():
            # File to keep latest epoch
            with open(os.path.join(args.save_path, 'first_run.txt'), 'w') as f:
                f.write(str(epoch + 1))
        self.ckpt_res_state_tracker.record_lowest_loss_epoch(cur_loss, epoch+1)
        if not with_eval_res:
            dummy=True
        else:
            dummy = False
            self.ckpt_res_state_tracker.record_bestF1_epoch(
                eval_stats[0], epoch+1)
            # log_eval_stats(eval_stats, self.logger)
        ckpt_res_stat_dict = self.ckpt_res_state_tracker.get_to_save_state_info(dummy=dummy)
        state_dict.update(ckpt_res_stat_dict)
        self.save_checkpoint(
            state_dict, 
            self.ckpt_res_state_tracker.to_copy,
            epoch+1)
