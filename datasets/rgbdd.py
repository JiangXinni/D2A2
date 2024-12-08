import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2
import glob
import skimage


class RGBDD_dataset(Dataset):
    """RGBDD  Dataset. Only for test. """

    def __init__(self, root_dir, scale=4):
        self.root_dir = root_dir
        self.scale = scale
        self.image_files = sorted(glob.glob(os.path.join(root_dir, "RGB", "*.jpg")))
        self.depth_files = sorted(glob.glob(os.path.join(root_dir, "Depth", "*.png")))#图是黑的，所以是原值
        self.mde_files = sorted(glob.glob(os.path.join(root_dir, "MDE_relative", "*.png")))
        self.size = len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        depth_file = self.depth_files[idx]
        mde_file = self.mde_files[idx]
        image = cv2.imread(image_file).astype(np.float32)   # [H, W, 3]
        depth = skimage.io.imread(depth_file).astype(np.float32) # [H, W]
        depth = depth / 1000.

        mde=cv2.imread(mde_file)[:,:,0].astype(np.float32) # [H, W]
        
        h, w = depth.shape
        s = self.scale
        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC))
        # normalize
        image = image.transpose(2,0,1) / 255.
        depth_min = depth.min()
        depth_max = depth.max()
        target = (target - depth_min) / (depth_max - depth_min)
        depth = (depth - depth_min) / (depth_max - depth_min)

        mde_min = mde.min()
        mde_max = mde.max()
        mde = (mde - mde_min) / (mde_max - mde_min)

        # to tensor
        image = torch.from_numpy(image).float()
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        target = torch.from_numpy(target).unsqueeze(0).float()
        mde = torch.from_numpy(mde).unsqueeze(0).float()
        
        sample = {'guidance': image, 'target': target, 'gt': depth, 'min': depth_min * 100,
                'max': depth_max * 100,'mde':mde}
        
        return sample

    def __len__(self):
        return self.size

