import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
from importlib import import_module
import torch.nn.functional as F
import functools
from utils import *
from datasets import *
from option import args

#choice the model
if args.scale==4:
    from models.D2A2_depthanything import D2A2
elif args.scale==8:
    from models.D2A2_depthanything_L_scale8 import D2A2
elif args.scale==16:
    from models.D2A2_depthanything_L_scale16 import D2A2




import cv2
import glob
import matplotlib
import numpy as np


setup_seed(20)
def mask_normal(tensor,mean=10):
    mask = tensor < tensor.mean()/mean
    tensor[mask] = (tensor[mask]-tensor[mask].min())/(tensor[mask].max()-tensor[mask].min())
    tensor[~mask] = 1
    return tensor

class MaskLoss(torch.nn.Module):
    def __init__(self,scale):
        super().__init__()
        self.scale=scale
        self.alpha = torch.nn.Parameter(torch.tensor(1.5))
        self.gamma = torch.nn.Parameter(torch.tensor(1.0))
        self.weig_func = lambda x, y, z: torch.exp((x-x.min()) / (x.max()-x.min()) * y) * z
        self.weig_func1 = lambda x, y, z: (x - x.min()) / (x.max() - x.min()) * y * z
        self.weig_func2 = lambda x, y, z: torch.exp(x * y) * z
        self.upscale_func = functools.partial(F.interpolate, mode='bicubic', align_corners=False)
        self.downscale_func = functools.partial(F.interpolate, mode='bicubic', align_corners=False)

    def forward(self,out,gt):

        H,W=gt.size()[2:]
        gt_up=self.upscale_func(gt, size=(H*self.scale,W*self.scale))
        gt_up_down=self.downscale_func(gt_up, size=gt.size()[2:])
        diff=torch.abs(gt - gt_up_down)
        diff_normal=mask_normal(diff)
        weight = self.weig_func2(diff_normal, self.alpha, self.gamma).detach()
        loss = torch.mean(weight* torch.abs(gt-out))
        return loss


model = D2A2(args).cuda()
print(model)
n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
if n_gpus > 1:
    model = torch.nn.DataParallel(model)



# dataloader
if args.dataset == 'nyu':
    train_dataset = NYU_v2_dataset(root_dir=args.dataset_dir, scale=args.scale,
                            augment=args.augment, input_size = args.input_size)
    test_dataset = NYU_v2_dataset(root_dir=args.dataset_dir, scale=args.scale,
                            train=False, augment=False, input_size=None)
else:
    raise NotImplementedError(f'Dataset {args.dataset} not found')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
print("dataloader done")


# trainer
if args.pretrain_path != None:
    model.load_state_dict(torch.load(args.pretrain_path))
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5,last_epoch= args.last_epoch)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5,last_epoch= args.last_epoch)

if args.loss is 'L1':
    criterion = torch.nn.L1Loss() #choice criterion
else:
    criterion = MaskLoss(args.scale)
trainer = Trainer(args, model, optimizer, scheduler, criterion, train_loader, test_loader)
print("trainer done")


### main
trainer.train()

