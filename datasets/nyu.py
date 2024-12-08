import numpy as np
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2

from datasets import common


class NYU_v2_dataset(Dataset):
    """NYUv2Dataset."""

    def __init__(self, root_dir, scale=4, train=True, augment=True,input_size = None):
        self.input_size = input_size
        self.root_dir = root_dir
        self.scale = scale
        self.augment = augment
        self.train = train
        if self.train:
            self.size  = 1000
        else:
            self.size = 449

    def __getitem__(self, idx):
        if not self.train:
            idx += 1000
        image_file = os.path.join(self.root_dir,'RGB',f'{idx}.jpg')
        depth_file = os.path.join(self.root_dir,'Depth',f'{idx}.npy')
        mde_file = os.path.join(self.root_dir, 'MDE_relative', f'{idx}.png')#add
        
        image = cv2.imread(image_file).astype(np.float32) # [H, W, 3]
        depth = np.load(depth_file) # [H, W]
        mde=cv2.imread(mde_file)[:,:,0].astype(np.float32)  # [H, W]


        if self.input_size is not None:
            x0 = random.randint(0, image.shape[0] - self.input_size)
            y0 = random.randint(0, image.shape[1] - self.input_size)
            image = image[x0:x0+self.input_size, y0:y0+self.input_size]
            depth = depth[x0:x0+self.input_size, y0:y0+self.input_size]
            mde=mde[x0:x0+self.input_size, y0:y0+self.input_size]#add
        
        h, w = depth.shape
        s = self.scale
        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC))
        # normalize
        image = image.transpose(2,0,1) / 255.

        depth_min = depth.min()
        depth_max = depth.max()
        target = (target - depth_min) / (depth_max - depth_min)
        depth = (depth - depth_min) / (depth_max - depth_min)

        mde_min=mde.min()
        mde_max=mde.max()
        mde=(mde-mde_min)/(mde_max-mde_min)
        # to tensor
        image = torch.from_numpy(image).float()
        mde = torch.from_numpy(mde).unsqueeze(0).float()
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        target = torch.from_numpy(target).unsqueeze(0).float()

        if self.augment:
            image, target, depth,mde = common.augment_info(image, target, depth,mde)#add
        
        sample = {'guidance': image, 'target': target, 'gt': depth, 'min': depth_min * 100,
                'max': depth_max * 100,'mde':mde}#add
        return sample


    def __len__(self):
        return self.size
