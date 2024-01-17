import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Deformable_Convolution_V2.modules.modulated_deform_conv import ModulatedDeformConv_Sep as DeformConv


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)

def convdown(in_channels, out_channels, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride)

class LTE(torch.nn.Module):
    def __init__(self, n_feats=64):
        super(LTE, self).__init__()
        
        self.conv_head = torch.nn.Sequential(
            conv3x3(3, n_feats), 
            nn.ReLU(inplace=True)
        )
        self.slice1 = torch.nn.Sequential(
            conv3x3(n_feats, n_feats), 
            nn.ReLU(inplace=True),
            conv3x3(n_feats, n_feats),
        )
        self.slice2 = torch.nn.Sequential(
            convdown(n_feats, n_feats),
            nn.ReLU(inplace=True),
            conv3x3(n_feats, n_feats)
        )
        self.slice3 = torch.nn.Sequential(
            convdown(n_feats, n_feats),
            nn.ReLU(inplace=True),
            conv3x3(n_feats, n_feats)
        )
    def forward(self, x):
        x = self.conv_head(x)
        x = self.slice1(x)
        x_lv1 = x
        x = self.slice2(x)
        x_lv2 = x
        x = self.slice3(x)
        x_lv3 = x
        return x_lv1, x_lv2, x_lv3


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res_scale=1.):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        
    def forward(self, x):
        x_ = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out.mul(self.res_scale) + x_
        return out


class SFE(nn.Module):
    def __init__(self, in_channels, n_feats, n_resblocks, res_scale=1.):
        super(SFE, self).__init__()
        self.n_resblocks = n_resblocks
        self.conv_head = conv3x3(in_channels, n_feats)
        self.RBs = nn.ModuleList()
        for _ in range(self.n_resblocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, 
                res_scale=res_scale))
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.act(self.conv_head(x))
        x_ = x
        for i in range(self.n_resblocks):
            x = self.RBs[i](x)
        x = x + x_
        return x



################  DDA Blocks  ################
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=64):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(True),
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(True),
            nn.Linear(mid_channels, out_channels)
        )
    def forward(self, x):
        return self.mlp(x)


def calc_mean_std(x, eps=1e-5):
    N, C = x.size()[:2]
    x_var = x.view(N, C, -1).var(dim=2) + eps
    x_std = x_var.sqrt().view(N, C, 1, 1)
    x_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return x_mean, x_std


class calc_mean_std_learnable(nn.Module):
    def __init__(self, in_feat, out_feat, eps=1e-5):
        super(calc_mean_std_learnable, self).__init__()
        self.eps = eps
        self.std_mlp = MLP(in_feat, out_feat)
        self.mean_mlp = MLP(in_feat, out_feat)
        self.out_feat = out_feat
    def forward(self, x):
        N, C = x.size()[:2]
        x_var = x.view(N, C, -1).var(dim=2) + self.eps
        x_std = x_var.sqrt()
        x_mean = x.view(N, C, -1).mean(dim=2)
        
        x_std = self.std_mlp(x_std).view(N, self.out_feat, 1, 1)
        x_mean = self.mean_mlp(x_mean).view(N, self.out_feat, 1, 1)

        return x_mean, x_std


class LearnableDomainAlignment(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(LearnableDomainAlignment, self).__init__()
        self.calc_msl = calc_mean_std_learnable(in_feat, out_feat)

    def forward(self, rgb, depth):
        size = rgb.size()
        depth_mean, depth_std = self.calc_msl(depth)
        rgb_mean, rgb_std = calc_mean_std(rgb)

        rgb_normalized = (rgb - rgb_mean.expand(
            size)) / rgb_std.expand(size)
        return rgb_normalized * depth_std.expand(size) + depth_mean.expand(size)
##############################################







################  MFA Blocks  ################
class GatedConv2dWithActivation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class PA(nn.Module):
    def __init__(self, n_feats):
        super(PA, self).__init__()
        self.mul_conv1 = nn.Conv2d(n_feats*2, n_feats, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        cat_input = torch.cat((feature_maps, para_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        return feature_maps * mul
##############################################




class Fusion(nn.Module):
    def __init__(self, n_feats):
        super(Fusion, self).__init__()
        # DDA
        self.LDA = LearnableDomainAlignment(n_feats, 1)
        self.conv_after_LDA = conv1x1(n_feats,n_feats)
        self.conv_fuse1 = conv3x3(n_feats*2,n_feats)
        self.get_offset = conv3x3(n_feats,n_feats)
        self.DCN = DeformConv(n_feats, n_feats, kernel_size=3, stride=1, padding=1,
                                          dilation=1,
                                          groups=1,
                                          deformable_groups=8, im2col_step=1)
        # MFA
        self.conv_fuse2 = conv3x3(n_feats*2,n_feats)
        self.GC = GatedConv2dWithActivation(n_feats, n_feats, 3, 1, padding=1)
        self.attention = PA(n_feats)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        
    def forward(self, depth, rgb):
        # LDA
        rgb_LDA = self.LDA(rgb, depth)  
        rgb_LDA = self.conv_after_LDA(rgb_LDA)
        rgb_LDA  = rgb + rgb_LDA
        # DGA
        offset = self.act(self.conv_fuse1(torch.cat((depth,rgb_LDA),dim = 1)))
        offset = self.act(self.get_offset(offset))
        rgb_guided = self.DCN(rgb, offset)
        # GC
        feature = self.conv_fuse2(torch.cat((depth, rgb_guided), dim=1))
        feature = self.GC(feature)
        # PA
        res = self.attention(depth, feature)
        
        return res



class Upsampler(nn.Module):
    def __init__(self, scale, n_feats):
        super(Upsampler, self).__init__()
        self.conv = conv3x3(n_feats, n_feats*scale*scale)
        self.up = nn.PixelShuffle(scale)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x





class MainNet(nn.Module):
    def __init__(self, n_feats, n_resblocks, res_scale=1.):
        super(MainNet, self).__init__()
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.SFE  = SFE(1, n_feats, n_resblocks, res_scale)

        self.conv11_head = conv3x3(n_feats, n_feats)
        self.concat1 = Fusion(n_feats)
        self.upsampler12 = Upsampler(2, n_feats)

        self.conv22_head = conv3x3(n_feats, n_feats)
        self.concat2 = Fusion(n_feats)
        self.upsampler23 = Upsampler(2, n_feats)

        self.conv33_head = conv3x3(n_feats, n_feats)
        self.concat3 = Fusion(n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats//2)
        self.conv_tail2 = conv1x1(n_feats//2, 1)


    def forward(self, x, ref_lv3=None, ref_lv2=None,ref_lv1=None):
        x_inter = F.interpolate(x, scale_factor=4, mode='bicubic')

        x = self.act(self.SFE(x))
        x11 = x

        ref_lv3 = self.act(self.conv11_head(ref_lv3))
        x11 = self.concat1(x11,ref_lv3)
        x22 = self.upsampler12(x11)

        ref_lv2 = self.act(self.conv22_head(ref_lv2))
        x22 = self.concat2(x22,ref_lv2)
        x33 = self.upsampler23(x22)

        ref_lv1 = self.act(self.conv33_head(ref_lv1))
        x33 = self.concat3(x33,ref_lv1)

        x = self.conv_tail1(x33)
        x = self.act(x)
        x = self.conv_tail2(x)
        
        x = x + x_inter
        return x




class D2A2(nn.Module):
    def __init__(self, args):
        super(D2A2, self).__init__()
        self.args = args
        self.LTE = LTE(args.n_feats)
        self.mainNet = MainNet(args.n_feats, args.n_resblocks, args.res_scale)
        
    def forward(self, rgb = None,depth = None):
        ref, lr = rgb, depth
        if self.args.scale > 4:
            lr = F.interpolate(lr, scale_factor=self.args.scale//4, mode='bicubic')
        ref_lv1, ref_lv2, ref_lv3 = self.LTE(ref)
        sr = self.mainNet(lr, ref_lv3, ref_lv2, ref_lv1)
        
        return sr

