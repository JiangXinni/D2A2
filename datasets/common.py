import numpy as np
import torch
import random


def augment_info(guidance, target, gt,mde, flip_h=True, rot=True,flip_v = True):
    if random.random() < 0.5 and flip_h:
        target = torch.from_numpy(target.numpy()[:, :, ::-1].copy())
        guidance = torch.from_numpy(guidance.numpy()[:, :, ::-1].copy())
        gt = torch.from_numpy(gt.numpy()[:, :, ::-1].copy())
        mde = torch.from_numpy(mde.numpy()[:, :, ::-1].copy())

    if  random.random() < 0.5 and rot:
        target = torch.from_numpy(target.numpy()[:, ::-1, :].copy())
        guidance = torch.from_numpy(guidance.numpy()[:, ::-1, :].copy())
        gt = torch.from_numpy(gt.numpy()[:, ::-1, :].copy())
        mde = torch.from_numpy(mde.numpy()[:, ::-1, :].copy())

    if random.random() < 0.5 and flip_v:
        target = torch.FloatTensor(np.transpose(target.numpy(),(0,2,1)))
        guidance = torch.FloatTensor(np.transpose(guidance.numpy(),(0,2,1)))
        gt = torch.FloatTensor(np.transpose(gt.numpy(),(0,2,1)))
        mde = torch.FloatTensor(np.transpose(mde.numpy(),(0,2,1)))

    return guidance, target, gt, mde


def get_patch(img_in, img_tar, patch_size):
    ix = random.randrange(0, img_in.shape[0] - patch_size + 1)
    iy = random.randrange(0, img_in.shape[1] - patch_size + 1)

    img_in = img_in[iy:iy + patch_size, ix:ix + patch_size]
    img_tar = img_tar[iy:iy + patch_size, ix:ix + patch_size]
