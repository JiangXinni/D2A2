import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Deformable_Convolution_V2.modules.modulated_deform_conv import ModulatedDeformConv_Sep as DeformConv
import numbers
from einops import rearrange
from torchsummary import summary
from thop import profile
from option import args




class Restormer_CNN_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Restormer_CNN_block, self).__init__()
        self.embed = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.GlobalFeature = GlobalFeatureExtraction(dim=out_dim, num_heads=8)
        self.LocalFeature = LocalFeatureExtraction(dim=out_dim)
        self.FFN = nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1, bias=False,
                             padding_mode="reflect")

    def forward(self, x):
        x = self.embed(x)
        x1 = self.GlobalFeature(x)
        x2 = self.LocalFeature(x)
        out = self.FFN(torch.cat((x1, x2), 1))
        return out


class GlobalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(GlobalFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim, out_fratures=dim,
                       ffn_expansion_factor=ffn_expansion_factor, )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LocalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim=64,
                 num_blocks=2,
                 ):
        super(LocalFeatureExtraction, self).__init__()
        self.Extraction = nn.Sequential(*[ResBlock(dim, dim) for i in range(num_blocks)])

    def forward(self, x):
        return self.Extraction(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                      padding_mode="reflect"),
        )

    def forward(self, x):
        out = self.conv(x)
        return out + x


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 out_fratures,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias, padding_mode="reflect")

        self.project_out = nn.Conv2d(
            hidden_features, out_fratures, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

################################################################


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
            conv3x3(4, n_feats),
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
        self.slice4 = torch.nn.Sequential(
            convdown(n_feats, n_feats),
            nn.ReLU(inplace=True),
            conv3x3(n_feats, n_feats)
        )
        self.slice5 = torch.nn.Sequential(
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
        x = self.slice4(x)
        x_lv4 = x
        x = self.slice5(x)
        x_lv5 = x
        return x_lv1, x_lv2, x_lv3,x_lv4, x_lv5


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


class SFTLayer(nn.Module):
    def __init__(self, n_feats=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(n_feats, n_feats, 1)
        self.SFT_scale_conv1 = nn.Conv2d(n_feats, 2 * n_feats, 1)
        self.SFT_shift_conv0 = nn.Conv2d(n_feats, n_feats, 1)
        self.SFT_shift_conv1 = nn.Conv2d(n_feats, 2 * n_feats, 1)

    def forward(self, feature, condition):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(condition), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(condition), 0.1, inplace=True))
        return feature * (scale + 1) + shift




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

        self.conv44_head = conv3x3(n_feats, n_feats)
        self.concat4 = Fusion(n_feats)
        self.upsampler34 = Upsampler(2, n_feats)

        self.conv55_head = conv3x3(n_feats, n_feats)
        self.concat5 = Fusion(n_feats)
        self.upsampler45 = Upsampler(2, n_feats)

        self.conv_tail1 = conv3x3(n_feats, n_feats//2)
        self.conv_tail2 = conv1x1(n_feats//2, 1)



        self.MDE_Encoder1 = Restormer_CNN_block(1, n_feats)
        self.MDE_Encoder2 = Restormer_CNN_block(n_feats, n_feats )
        self.MDE_Encoder3 = Restormer_CNN_block(n_feats, n_feats )
        self.MDE_Encoder4 = Restormer_CNN_block(n_feats, n_feats)
        self.MDE_Encoder5 = Restormer_CNN_block(n_feats, n_feats)

        self.concatMDE1=  Fusion(n_feats)
        self.concatMDE2 = Fusion(n_feats)
        self.concatMDE3 = Fusion(n_feats)
        self.concatMDE4 = Fusion(n_feats)
        self.concatMDE5 = Fusion(n_feats)


        self.condition_convdown1= nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        self.condition_convdown2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        self.condition_convdown3 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        self.condition_convdown4 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")



    def forward(self, x,ref_lv5=None, ref_lv4=None,ref_lv3=None, ref_lv2=None,ref_lv1=None,MDE=None):
        x_inter = F.interpolate(x, scale_factor=16, mode='bicubic')
        mde_condition1=self.MDE_Encoder1(MDE)
        mde_condition2=self.MDE_Encoder2(self.condition_convdown1(mde_condition1))
        mde_condition3=self.MDE_Encoder3(self.condition_convdown2(mde_condition2))
        mde_condition4 = self.MDE_Encoder4(self.condition_convdown3(mde_condition3))
        mde_condition5 = self.MDE_Encoder5(self.condition_convdown4(mde_condition4))


        x = self.act(self.SFE(x))
        x11 = x

        ref_lv5 = self.act(self.conv11_head(ref_lv5))
        x11 = self.concatMDE1(x11, mde_condition5)
        x11 = self.concat1(x11,ref_lv5)
        x22 = self.upsampler12(x11)

        ref_lv4 = self.act(self.conv22_head(ref_lv4))
        x22 = self.concatMDE2(x22, mde_condition4)
        x22 = self.concat2(x22,ref_lv4)
        x33 = self.upsampler23(x22)

        ref_lv3 = self.act(self.conv33_head(ref_lv3))
        x33 = self.concatMDE3(x33, mde_condition3)
        x33 = self.concat3(x33,ref_lv3)
        x44 = self.upsampler34(x33)

        ref_lv2 = self.act(self.conv44_head(ref_lv2))
        x44 = self.concatMDE4(x44, mde_condition2)
        x44 = self.concat4(x44,ref_lv2)
        x55 = self.upsampler45(x44)

        ref_lv1 = self.act(self.conv55_head(ref_lv1))
        x55 = self.concatMDE5(x55, mde_condition1)
        x55 = self.concat5(x55, ref_lv1)

        x = self.conv_tail1(x55)
        x = self.act(x)
        x = self.conv_tail2(x)


        x=x+x_inter
        return x




class D2A2(nn.Module):
    def __init__(self, args):
        super(D2A2, self).__init__()
        self.args = args
        self.LTE = LTE(args.n_feats)
        self.mainNet = MainNet(args.n_feats, args.n_resblocks, args.res_scale)

    def forward(self, rgb = None,depth = None,MDE=None):
        ref, lr = rgb, depth


        ref=torch.cat([ref, MDE], dim=1)

        ref_lv1, ref_lv2, ref_lv3,ref_lv4,ref_lv5 = self.LTE(ref)


        sr = self.mainNet(lr, ref_lv5,ref_lv4,ref_lv3, ref_lv2, ref_lv1,MDE)

        return sr


if __name__ == '__main__':


    model = D2A2(args).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params}")


    img = torch.randn(1, 3, 256, 256).cuda()
    depth = torch.randn(1, 1, 16, 16).cuda()
    mde=torch.randn(1,1,256,256).cuda()
    out=model(img,depth,mde)

    print(out.shape)

    flops, params = profile(model, inputs=(img,depth,mde))
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")

