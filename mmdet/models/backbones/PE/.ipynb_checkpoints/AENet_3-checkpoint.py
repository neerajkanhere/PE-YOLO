import math

import cv2
import numpy as np
import torch
import torch.nn as nn
from ...builder import BACKBONES
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import BaseModule
from .DeformConv import ModulatedDeformableConv2d as DCN


# Upsample
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            CBL(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, x):
        x = self.upsample(x)
        return x

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')  # replicate    # reflect
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(in_features, out_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout) * x

  

@BACKBONES.register_module()
class AENet_3(nn.Module):
    def __init__(self,
                 num_high=3,
                 ch_blocks=32,
                 up_ksize=1,
                 high_ch=32,
                 high_ksize=3,
                 ch_mask=32,
                 gauss_kernel=5):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)
#         self.trans_low = Trans_low(ch_blocks, ch_mask)

#         for i in range(0, self.num_high):
#             self.__setattr__('up_guide_layer_{}'.format(i),
#                              Up_guide(up_ksize, ch=3))
#             self.__setattr__('trans_high_layer_{}'.format(i),
#                              Trans_high(3, high_ch, 3, high_ksize))

        for i in range(0, self.num_high+1):
        
            self.__setattr__('AE_{}'.format(i), AE(3))

    def forward(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(img=x)

        trans_pyrs = []
#         trans_pyr, guide = self.trans_low(pyrs[-1])
#         trans_pyrs.append(trans_pyr)

#         commom_guide = []
#         for i in range(self.num_high):
#             guide = self.__getattr__('up_guide_layer_{}'.format(i))(guide)
#             commom_guide.append(guide)

#         for i in range(self.num_high):
#             trans_pyr = self.__getattr__('trans_high_layer_{}'.format(i))(
#                 pyrs[-2 - i], commom_guide[i])
#             trans_pyrs.append(trans_pyr)

        for i in range(self.num_high+1):
            trans_pyr = self.__getattr__('AE_{}'.format(i))(
                pyrs[-1 - i])
            trans_pyrs.append(trans_pyr)
        out = self.lap_pyramid.pyramid_recons(trans_pyrs)
        
        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    
class DPM(nn.Module):
    def __init__(self, inplanes, planes, act=nn.LeakyReLU(negative_slope=0.2,inplace=True), bias=False):
        super(DPM, self).__init__()

        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias),
            act,
            nn.Conv2d(planes, inplanes, kernel_size=1, bias=bias)
        )

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term
        return x
  

import cv2
from torchvision import transforms
def sobel(img):

    x = img.squeeze(0).cpu().numpy().transpose(1,2,0)
    x = x*255
    x_x = cv2.Sobel(x, cv2.CV_64F, 1, 0)
    x_y = cv2.Sobel(x, cv2.CV_64F, 0, 1)
    add_x = cv2.addWeighted(x_x,0.5,x_y,0.5,0)
    add_x = transforms.ToTensor()(add_x).unsqueeze(0)
    return add_x

class AE(nn.Module):
    def __init__(self, n_feat, reduction=8, bias=False, act=nn.LeakyReLU(negative_slope=0.2,inplace=True), groups =1):

        super(AE, self).__init__()

        self.n_feat = n_feat
        self.groups = groups
        self.reduction = reduction
        self.agg = nn.Conv2d(6,
                      3,
                      1,
                      stride=1,
                      padding=0,
                      bias=False)
        self.conv_edge = nn.Conv2d(3, 3, kernel_size=1, bias=bias)
        
        self.res1 = ResidualBlock(3, 32)
        self.res2 = ResidualBlock(32, 3)
        self.dpm = nn.Sequential(DPM(32, 32))
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=1)
        self.lpm = LowPassModule(32)
        self.fusion = nn.Conv2d(6, 3, kernel_size=1)
#         self.prior_sp = Prior_Sp(3)
        
    def forward(self, x):
        s_x = sobel(x)
        s_x = self.conv_edge(x)
        
        res = self.res1(x)
        res = self.dpm(res)
        res = self.res2(res)
        out = torch.cat([res, s_x+x],dim=1)
        out = self.agg(out)
        
        low_fea = self.conv1(x)
        low_fea = self.lpm(low_fea)
        low_fea = self.conv2(low_fea)
        
        out = torch.cat([out, low_fea], dim=1)
        out = self.fusion(out)
        
        return out

class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch =  in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1)
        priors = [F.upsample(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear') for i in range(4)]
        bottle = torch.cat(priors, 1)
        
        return self.relu(bottle)
    
# class RSABlock(nn.Module):

#     def __init__(self, input_channel=3, output_channel=3, offset_channel=3):
#         super().__init__()
#         self.in_channel = input_channel
#         self.out_channel = output_channel
#         if self.in_channel != self.out_channel:
#             self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
#         self.dcnpack = DCN(output_channel, output_channel, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
#                             extra_offset_mask=True, offset_in_channel=offset_channel)
#         self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         self.initialize_weights()

#     def forward(self, x, offset):
#         if self.in_channel != self.out_channel:
#             x = self.conv0(x)
#         fea = self.lrelu(self.dcnpack([x, offset]))
#         out = self.conv1(fea) + x
#         return out
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.xavier_uniform_(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.zero_()

# class Prior_Sp(nn.Module):
#     """ Channel attention module"""
#     def __init__(self, in_dim=32):
#         super(Prior_Sp, self).__init__()
#         self.chanel_in = in_dim

#         self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
#         self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

#         self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
#         # self.gamma1 = nn.Parameter(torch.zeros(1))
#         self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
#         # self.softmax  = nn.Softmax(dim=-1)
#         self.sig = nn.Sigmoid()
#     def forward(self,x, prior):
        
#         x_q = self.query_conv(x)
#         prior_k = self.key_conv(prior)
#         energy = x_q * prior_k
#         attention = self.sig(energy)
#         # print(attention.size(),x.size())
#         attention_x = x * attention
#         attention_p = prior * attention

#         x_gamma = self.gamma1(torch.cat((x, attention_x),dim=1))
#         x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

#         p_gamma = self.gamma2(torch.cat((prior, attention_p),dim=1))
#         prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

#         return x_out, prior_out                    
    
# eanet = EANet()

# im = torch.ones(3,256,256).unsqueeze(0)

# print(eanet(im).shape)