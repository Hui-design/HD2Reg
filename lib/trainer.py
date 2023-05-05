import time, os, torch,copy
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from lib.timer import Timer, AverageMeter
from lib.utils import Logger,validate_gradient

from tqdm import tqdm
import torch.nn.functional as F
import gc
import pdb
import time


class Trainer(object):
    def __init__(self, args):
        self.config = args
        # parameters
        self.start_epoch = 1
        self.max_epoch = args.max_epoch
        self.save_dir = args.save_dir
        self.device = args.device
        self.verbose = args.verbose
        self.max_points = args.max_points

        self.model = args.model.to(self.device)
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_freq = args.scheduler_freq
        self.snapshot_freq = args.snapshot_freq
        self.snapshot_dir = args.snapshot_dir 
        self.benchmark = args.benchmark
        self.iter_size = args.iter_size
        self.verbose_freq= args.verbose_freq

        self.w_circle_loss = args.w_circle_loss
        self.w_overlap_loss = args.w_overlap_loss
        self.w_saliency_loss = args.w_saliency_loss 
        self.loss = args.desc_loss

        self.best_loss = 1e5
        self.best_recall = -1e5
        self.writer = SummaryWriter(log_dir=args.tboard_dir)
        self.logger = Logger(args.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()])/1000000.} M\n')
        

        if (args.pretrain !=''):
            self._load_pretrain(args.pretrain)
        
        self.loader =dict()
        self.loader['train']=args.train_loader
        self.loader['val']=args.val_loader
        self.loader['test'] = args.test_loader

        with open(f'{args.snapshot_dir}/model','w') as f:
            f.write(str(self.model))
        f.close()
 
    def _snapshot(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_recall': self.best_recall
        }
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
        self.logger.write(f"Save model to {filename}\n")
        torch.save(state, filename)

    def _load_pretrain(self, resume):
        if os.path.isfile(resume):
            state = torch.load(resume)
            self.model.load_state_dict(state['state_dict'])
            self.start_epoch = state['epoch']
            self.scheduler.load_state_dict(state['scheduler'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.best_loss = state['best_loss']
            self.best_recall = state['best_recall']
            
            self.logger.write(f'Successfully load pretrained model from {resume}!\n')
            self.logger.write(f'Current best loss {self.best_loss}\n')
            self.logger.write(f'Current best recall {self.best_recall}\n')
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def stats_dict(self):
        stats=dict()
        stats['loss'] = 0.
        stats['overlap_loss'] = 0.
        stats['low_feat_loss'] = 0.
        stats['high_feat_loss'] = 0.
        stats['l_saliency_loss'] = 0.
        stats['h_saliency_loss'] = 0.
        stats['low_feat_recall'] = 0.
        stats['high_feat_recall'] = 0.
        stats['overlap_precision'] = 0.
        # stats['l_saliency_precision'] = 0.
        # stats['h_saliency_precision'] = 0.
        return stats

    def stats_meter(self):
        meters=dict()
        stats=self.stats_dict()
        for key,_ in stats.items():
            meters[key]=AverageMeter()
        return meters


    def inference_one_batch(self, inputs, phase):
        assert phase in ['train','val','test']
        ##################################
        # training
        if(phase == 'train'):
            self.model.train()
            ###############################################
            # forward pass
            # pdb.set_trace()
            l_feats, h_feats, l_saliency, h_saliency, scores_overlap = self.model(inputs)
            pcd = inputs['points'][0]
            len_src = inputs['stack_lengths'][0][0]
            c_rot, c_trans = inputs['rot'], inputs['trans']
            correspondence = inputs['correspondences']
            src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
            # pdb.set_trace()
            ###################################################
            # get loss
            stats = self.loss(src_pcd, tgt_pcd, l_feats, h_feats, scores_overlap, len_src,
                correspondence, c_rot, c_trans, l_saliency, h_saliency)

            # w_saliency_loss turn to 1 after the features are well trained.
            loss = 0.5*stats['low_feat_loss'] + stats['high_feat_loss'] + stats['overlap_loss'] \
                    + self.w_saliency_loss * (stats['l_saliency_loss'] + stats['h_saliency_loss'])
                   
            stats['loss'] = loss
            loss.backward()

        else:
            self.model.eval()
            with torch.no_grad():
                ###############################################
                # forward pass
                l_feats, h_feats, l_saliency, h_saliency, scores_overlap = self.model(inputs)
                pcd = inputs['points'][0]
                len_src = inputs['stack_lengths'][0][0]
                c_rot, c_trans = inputs['rot'], inputs['trans']
                correspondence = inputs['correspondences']
                src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
                # pdb.set_trace()
                ###################################################
                # get loss
                stats = self.loss(src_pcd, tgt_pcd, l_feats, h_feats, scores_overlap, len_src,
                    correspondence, c_rot, c_trans, l_saliency, h_saliency)
                loss = 0.5*stats['low_feat_loss'] + stats['high_feat_loss'] + stats['overlap_loss'] \
                    + self.w_saliency_loss * (stats['l_saliency_loss'] + stats['h_saliency_loss']) 
                stats['loss'] = loss

        ##################################
        # detach the gradients for loss terms
        stats['overlap_loss'] = float(stats['overlap_loss'].detach())
        stats['low_feat_loss'] = float(stats['low_feat_loss'].detach())
        stats['high_feat_loss'] = float(stats['high_feat_loss'].detach())
        stats['l_saliency_loss'] = float(stats['l_saliency_loss'].detach())
        stats['h_saliency_loss'] = float(stats['h_saliency_loss'].detach())
        stats['loss'] = float(stats['loss'].detach())

        return stats


    def inference_one_epoch(self,epoch, phase):
        gc.collect()
        assert phase in ['train','val','test']

        # init stats meter
        stats_meter = self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
        c_loader_iter = self.loader[phase].__iter__()
        
        self.optimizer.zero_grad()
        # for c_iter in tqdm(range(num_iter)): # loop through this epoch   
        self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '\n')  
        for c_iter in tqdm(range(num_iter)):
            ##################################
            # load inputs to device.
            inputs = c_loader_iter.next()
            for k, v in inputs.items():  
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                elif type(v) == dict:
                    pass
                else:
                    inputs[k] = v.to(self.device)
            # try:
            ##################################
            # forward pass
            # with torch.autograd.detect_anomaly():
            stats = self.inference_one_batch(inputs, phase)
            
            ###################################################
            # run optimisation
            if((c_iter+1) % self.iter_size == 0 and phase == 'train'):
                gradient_valid = validate_gradient(self.model)
                if(gradient_valid):
                    self.optimizer.step()
                else:
                    self.logger.write('gradient not valid\n')
                self.optimizer.zero_grad()
            
            ################################
            # update to stats_meter
            for key,value in stats.items():
                stats_meter[key].update(value)
            # except Exception as inst:
            #     print(inst)
            
            torch.cuda.empty_cache()
            
            if (c_iter + 1) % self.verbose_freq == 0 and self.verbose:  
                message = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' '
                curr_iter = num_iter * (epoch - 1) + c_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)
                
                message += f'{phase} Epoch: {epoch} [{c_iter+1:4d}/{num_iter}]'
                for key,value in stats_meter.items():
                    message += f'{key}: {value.avg:.2f}\t'

                self.logger.write(message + '\n')

        message = f'{phase} Epoch: {epoch}'
        for key,value in stats_meter.items():
            message += f'{key}: {value.avg:.2f}\t'
        self.logger.write(message+'\n')

        return stats_meter


    def train(self):
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            self.inference_one_epoch(epoch,'train')
            self.scheduler.step()
            
            stats_meter = self.inference_one_epoch(epoch,'val')
            
            if stats_meter['high_feat_loss'].avg < self.best_loss:
                self.best_loss = stats_meter['high_feat_loss'].avg
                self._snapshot(epoch, 'best_loss')

            if stats_meter['high_feat_recall'].avg > self.best_recall:
                self.best_recall = stats_meter['high_feat_recall'].avg
                self._snapshot(epoch, 'best_recall')
            
            # we only add saliency loss when we get descent point-wise features
            if(stats_meter['high_feat_recall'].avg>0.3):
                self.w_saliency_loss = 1.
                self.logger.write('w_saliency turns to 1.0')
            else:
                self.w_saliency_loss = 0.
                    
        # finish all epoch
        print("Training finish!")


    def eval(self):
        print('Start to evaluate on validation datasets...')
        stats_meter = self.inference_one_epoch(0,'val')
        
        for key, value in stats_meter.items():
            print(key, value.avg)
