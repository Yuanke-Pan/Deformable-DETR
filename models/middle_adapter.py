import torch
import torch.nn.functional as F
from torch import nn
from util.misc import NestedTensor

class UpSampling(nn.Module):

    def __init__(self, in_channels):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(in_channels, in_channels // 2, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(x)
        return x

class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        out_channels = in_channels // 2    
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.double_conv(x)
        return x
    
class Up_Conv(nn.Module):
    def __init__(self, in_channels, have_pre: bool, need_up: bool) -> None:
        super().__init__()
        self.need_up = need_up
        if need_up:
            self.UpSampling = UpSampling(in_channels)
        if have_pre:
            self.DoubleConv = DoubleConv(in_channels * 2)
    
    def forward(self, pre, x):
        if pre != None:
            pre = F.interpolate(pre, size=x.shape[-2:], mode="nearest")
            pre = torch.cat([pre, x], dim=1)
            pre = self.DoubleConv(pre)
        else:
            pre = x
        if self.need_up:
            pre = self.UpSampling(pre)
        return pre

        
class FeatureFusionBlock(nn.Module):
    
    def __init__(self, backbone, num_feature_levels) -> None:
        super().__init__()
        self.num_backbone_outs = len(backbone.strides)
        self.feature_addition_list = []
        self.feature_fusion_list = []
        num_channels = backbone.num_channels
        input_channels = backbone.num_channels[-1]
        for _ in range(num_feature_levels - self.num_backbone_outs):
            self.feature_addition_list.append(nn.Sequential(
                nn.Conv2d(input_channels, 2 * input_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, 2 * input_channels),
                nn.ReLU()
            ))
            input_channels = 2 * input_channels
            num_channels.append(input_channels)
        self.DownSampe = nn.Sequential(
            nn.Conv2d(num_channels[0], num_channels[0] // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.feature_addition_list = nn.ModuleList(self.feature_addition_list)
        for _ in range(len(num_channels)):
            self.feature_fusion_list.append(Up_Conv(num_channels[_], _ != (len(num_channels) - 1), _ != 0))
        self.feature_fusion_list = nn.ModuleList(self.feature_fusion_list)
        
    
    def forward(self, features):
        t = features[-1].tensors
        for block in self.feature_addition_list:
            z = block(t)
            features.append(NestedTensor(z, None))
            t = z
        t = None
        for i in range(-1, -len(features)-1, -1):
            #print(features[i].tensors)
            t = self.feature_fusion_list[i](t, features[i].tensors)
        t = self.DownSampe(t)
        return [NestedTensor(t, features[0].mask)]

        
        