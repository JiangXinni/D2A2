import os
import torch.nn as nn
import random
import torch
import numpy as np
import cv2
import argparse

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt
import wandb
import time
import torch.nn.functional as F
import functools

total_time = 0.0



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def calc_rmse(a, b, min, max):
    a = a * (max - min) + min
    b = b * (max - min) + min
    return np.sqrt(np.mean(np.power(a - b, 2)))



class Trainer(object):
    def __init__(self, args, model, optimizer, scheduler, criterion, train_loader, test_loader):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.maxepoch = args.epoch
        self.nowepoch = 0
        self.upscale_func = functools.partial(
            F.interpolate, mode='bicubic', align_corners=False
        )


        # prepare log
        s = datetime.now().strftime('%Y%m%d%H%M%S')
        result_root = '%s/%s-x%s-%s'%(args.trainresult, args.model_file, args.scale, s)
        os.makedirs(result_root, exist_ok=True)
        logging.basicConfig(filename='%s/train.log'%result_root,format='%(asctime)s %(message)s', level=logging.INFO)
        logging.info('args:\n%s\n'%(self.args))
        self.result_root = result_root


    @torch.no_grad()
    def validate(self):
        self.model.eval()
        rmse = np.zeros(self.test_loader.__len__())
        
        t = tqdm(iter(self.test_loader), leave=True, total=self.test_loader.__len__())
        for idx, data in enumerate(t):            
            guidance, target, gt, min, max,mde = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda(), data['min'], data['max'],data['mde'].cuda()
            
            out = self.model(guidance,target,mde)

            gt_ = gt[0,0].cpu().numpy()
            out_ = out[0,0].cpu().numpy()
            if self.args.dataset == "nyu":
                gt_ = gt_[6:-6, 6:-6]
                out_ = out_[6:-6, 6:-6]
            rmse[idx] = calc_rmse(gt_, out_, min.numpy(),max.numpy())
            
            t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
            t.refresh()
        return rmse


    def train(self):
        max_epoch = self.maxepoch
        best_rmse = 100.0
        best_epoch = 0

        for epoch in range(max_epoch):
            self.nowepoch = epoch + 1
            self.model.train()
            running_loss = 0.0
            
            t = tqdm(iter(self.train_loader), leave=True, total=self.train_loader.__len__())
            for idx, data in enumerate(t):
                self.optimizer.zero_grad()
                self.scheduler.step()
                guidance, target, gt,mde = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda(),data['mde'].cuda()
                out = self.model(guidance, target,mde)
                loss = self.criterion(out, gt)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.data.item()

                if idx % 50 == 0:
                    running_loss /= 50
                    t.set_description('[train epoch(L1):%d] loss: %.10f' % (self.nowepoch, running_loss))
                    t.refresh()
                    logging.info('epoch:%d running_loss:%.10f' % (self.nowepoch, running_loss))
            
            logging.info('epoch:%d optimizer_lr:%s' % (self.nowepoch, self.optimizer.state_dict()['param_groups'][0]['lr']))
            torch.save(self.model.state_dict(), "%s/last_parameter" % (self.result_root))

            if epoch%5==0:
                rmse = self.validate()
                mean_rmse = rmse.mean()
                logging.info('epoch:%d --------mean_rmse:%.10f ' % (self.nowepoch, mean_rmse))

                if mean_rmse < best_rmse:
                    best_rmse = mean_rmse
                    best_epoch = self.nowepoch
                    torch.save(self.model.state_dict(), "%s/best_parameter" % (self.result_root))
                logging.info('best_epoch:%d --------best_mean_rmse:%.10f ' % (best_epoch, best_rmse))
            if epoch >= max_epoch - 5:
                torch.save(self.model.state_dict(), f"%s/{epoch}_parameter" % (self.result_root))




class Tester(object):
    def __init__(self, args, model, test_loader):
        self.args = args
        self.model = model
        self.test_loader = test_loader

        # prepare log
        s = datetime.now().strftime('%Y%m%d%H%M%S')
        result_root = '%s/%s-%s-x%s-%s'%(args.testresult, args.model_file, args.dataset, args.scale, s)
        os.makedirs(result_root, exist_ok=True)
        if self.args.save:
            save_depth_root = result_root + "/depthsr"
            save_hotmap_root = result_root + "/hotmap"
            os.makedirs(save_depth_root, exist_ok=True)
            os.makedirs(save_hotmap_root, exist_ok=True)
            self.save_depth_root = save_depth_root
            self.save_hotmap_root = save_hotmap_root
        logging.basicConfig(filename='%s/test.log'%result_root,format='%(asctime)s %(message)s', level=logging.INFO)
        self.result_root = result_root


    @torch.no_grad()
    def validate(self):
        self.model.eval()
        rmse = np.zeros(self.test_loader.__len__())


        global total_time
        total_time = 0.0

        t = tqdm(iter(self.test_loader), leave=True, total=self.test_loader.__len__())
        for idx, data in enumerate(t):
            guidance, target, gt, min, max, mde = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda(), \
            data['min'], data['max'], data['mde'].cuda()

            begin_time = time.time()
            out = self.model(guidance, target, mde)
            end_time = time.time()
            total_time = total_time + (end_time - begin_time)

            errormap = torch.abs(gt - out)

            gt_ = gt[0, 0].cpu().numpy()
            out_ = out[0, 0].cpu().numpy()
            if self.args.dataset == "nyu":
                gt_ = gt_[6:-6, 6:-6]
                out_ = out_[6:-6, 6:-6]
            rmse[idx] = calc_rmse(gt_, out_, min.numpy(), max.numpy())

            if self.args.save:
                out_depth = out[0][0].cpu().numpy() * (max.numpy() - min.numpy()) + min.numpy()
                gt_depth = gt[0][0].cpu().numpy() * (max.numpy() - min.numpy()) + min.numpy()
                lr_depth = target[0][0].cpu().numpy() * (max.numpy() - min.numpy()) + min.numpy()
                errormap = errormap[0][0].cpu().numpy() * (max.numpy() - min.numpy()) + min.numpy()

                plt.imsave(os.path.join(self.save_hotmap_root, f'{idx}_sr_color.png'), out_depth, cmap='plasma')
                plt.imsave(os.path.join(self.save_hotmap_root, f'{idx}_errormap.png'), errormap, cmap='afmhot')
                plt.imsave(os.path.join(self.save_hotmap_root, f'{idx}_gt_color.png'), gt_depth, cmap='plasma')
                plt.imsave(os.path.join(self.save_hotmap_root, f'{idx}_lr_color.png'), lr_depth, cmap='plasma')

            t.set_description('[validate] rmse: %f' % rmse[:idx + 1].mean())

            t.refresh()
            logging.info('idx:%d rmse:%.10f' % (idx, rmse[idx]))

        logging.info('mean rmse:%.10f\n\n\n' % (rmse.mean()))

        logging.info("total_time:%.10f" % (total_time))


        return rmse





        


 