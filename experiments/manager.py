import numpy as np
import torch.distributed as dist

from experiments.gpu_utils import is_main_process
from utils import eval_3D_lane


class CkptResStateManager(object):
    def __init__(self, logger):
        # Logging setup
        self.best_epoch = 0
        self.lowest_loss = np.inf
        self.best_f1_epoch = 0
        self.best_val_f1 = -1e-5
        self.optim_saved_state = None
        self.schedule_saved_state = None
        self.to_copy = False
        self.cur_f1 = 0
        self.cur_loss = 1e5
        self.cur_epoch = -1
        self.logger = logger

    def update_all(self, ckpt_res_stat_dict):
        for k, v in ckpt_res_stat_dict.items():
            setattr(self, k, v)
    
    def record_lowest_loss_epoch(self, income_loss, income_epoch):
        self.cur_loss = income_loss
        self.cur_epoch = income_epoch
        if income_loss < self.lowest_loss:
            self.lowest_loss = income_loss
            self.best_epoch = income_epoch

    def record_bestF1_epoch(self, income_F1, income_epoch):
        self.cur_f1 = income_F1
        self.cur_epoch = income_epoch
        if income_F1 > self.best_val_f1:
            self.best_val_f1 = income_F1
            self.best_f1_epoch = income_epoch
            self.to_copy = True
        else:
            self.to_copy = False
        self.logger.info("===> Cur best F1 was {:.8f} in epoch {}".format(self.best_val_f1, self.best_f1_epoch))
        self.to_copy = False
    
    def get_all_ckpt_state_info(self):
        return self.__dict__

    def get_to_save_state_info(self, dummy=False):
        return dict(
            loss = self.cur_loss,
            f1 = 0 if dummy else self.cur_f1,
            best_epoch = self.cur_epoch if dummy else self.best_epoch,
            lowest_loss = 1e3 if dummy else self.lowest_loss,
            best_f1_epoch = self.cur_epoch if dummy else self.best_f1_epoch,
            best_val_f1 = 0 if dummy else self.best_val_f1
        )


class EvalManager(object):
    def __init__(self, args):
        self.evaluator = self._get_evaluator(args)
    
    @staticmethod
    def _get_evaluator(args):
        if 'openlane' in args.dataset_name:
            if 'v2' in args.dataset_name:
                evaluator = eval_3D_lane.LaneV2Eval(args)
            else:
                evaluator = eval_3D_lane.LaneEval(args)
        elif 'once' in args.dataset_name:
            evaluator = eval_3D_once.LaneEval()
        else:
            evaluator = eval_3D_lane.LaneEval(args)
        return evaluator
    
    def eval_pred(self, pred_lines, gt_lines, args, logger):
        # cal eval res
        if 'openlane' in args.dataset_name:
            eval_stats = self.evaluator.bench_one_submit_openlane_DDP(
                pred_lines, gt_lines, args.model_name, args.pos_threshold, vis=False)
        elif 'once' in args.dataset_name:
            eval_stats = self.evaluator.lane_evaluation(
                args.data_dir+'test', './data_splits/once/PersFormer/once_pred/test', args.eval_config_dir, 
                args)
        else:
            eval_stats = self.evaluator.bench_one_submit(pred_lines, gt_lines, vis=False)
        # gather gpus res
        if 'openlane' in args.dataset_name:
            eval_stats = EvalManager.cal_openlane_gpus_eval(args.world_size, eval_stats)
            # loss_list = []
        elif 'once' in args.dataset_name:
            # loss_list = []
            eval_stats = None
        else:
            raise NotImplementedError('only openlane & once supported yet.')
        # log eval_stats
        if is_main_process() and eval_stats != None:
            #TODO
            # log eval loss 
            # self.logger.info("===> Average {}-loss on validation set is {:.8f}".format(self.crit_string, loss_list[0].avg))
            log_eval_stats(eval_stats, logger) 
        return eval_stats

    @staticmethod
    def cal_openlane_gpus_eval(world_size, eval_stats):
        gather_output = [None for _ in range(world_size)]
        # all_gather all eval_stats and calculate mean
        dist.all_gather_object(gather_output, eval_stats)
        dist.barrier()
        r_lane = np.sum([eval_stats_sub[8] for eval_stats_sub in gather_output])
        p_lane = np.sum([eval_stats_sub[9] for eval_stats_sub in gather_output])
        c_lane = np.sum([eval_stats_sub[10] for eval_stats_sub in gather_output])
        cnt_gt = np.sum([eval_stats_sub[11] for eval_stats_sub in gather_output])
        cnt_pred = np.sum([eval_stats_sub[12] for eval_stats_sub in gather_output])
        match_num = np.sum([eval_stats_sub[13] for eval_stats_sub in gather_output])
        if cnt_gt!=0 :
            Recall = r_lane / cnt_gt
        else:
            Recall = r_lane / (cnt_gt + 1e-6)
        if cnt_pred!=0:
            Precision = p_lane / cnt_pred
        else:
            Precision = p_lane / (cnt_pred + 1e-6)
        if (Recall + Precision)!=0:
            f1_score = 2 * Recall * Precision / (Recall + Precision)
        else:
            f1_score = 2 * Recall * Precision / (Recall + Precision + 1e-6)
        if match_num!=0:
            category_accuracy = c_lane / match_num
        else:
            category_accuracy = c_lane / (match_num + 1e-6)
        eval_stats[0] = f1_score
        eval_stats[1] = Recall
        eval_stats[2] = Precision
        eval_stats[3] = category_accuracy
        eval_stats[4] = np.sum([eval_stats_sub[4] for eval_stats_sub in gather_output]) / world_size
        eval_stats[5] = np.sum([eval_stats_sub[5] for eval_stats_sub in gather_output]) / world_size
        eval_stats[6] = np.sum([eval_stats_sub[6] for eval_stats_sub in gather_output]) / world_size
        eval_stats[7] = np.sum([eval_stats_sub[7] for eval_stats_sub in gather_output]) / world_size

        return eval_stats

def log_eval_stats(eval_stats, logger):
    if is_main_process():
        logger.info("===> Evaluation laneline F-measure: {:.8f}".format(eval_stats[0]))
        logger.info("===> Evaluation laneline Recall: {:.8f}".format(eval_stats[1]))
        logger.info("===> Evaluation laneline Precision: {:.8f}".format(eval_stats[2]))
        logger.info("===> Evaluation laneline Category Accuracy: {:.8f}".format(eval_stats[3]))
        logger.info("===> Evaluation laneline x error (close): {:.8f} m".format(eval_stats[4]))
        logger.info("===> Evaluation laneline x error (far): {:.8f} m".format(eval_stats[5]))
        logger.info("===> Evaluation laneline z error (close): {:.8f} m".format(eval_stats[6]))
        logger.info("===> Evaluation laneline z error (far): {:.8f} m".format(eval_stats[7]))
