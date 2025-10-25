import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConv(nn.Module):
    # Multi-scale convolution module
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k//2) for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = sum([conv(x) for conv in self.convs])
        out = self.bn(out)
        return self.relu(out)

# class MSCAAttention(nn.Module):
#     def __int__(self,dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim,dim,5,padding=2,groups=dim)
#         self.conv0_1 = nn.Conv2d(dim,dim,(1,7),padding=(0,3),groups=dim)
#         self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
#
#         self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
#
#         self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
#         self.conv3 = nn.Conv2d(dim,dim,1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0(x)
#         attn_0 = self.conv0_1(attn)
#         attn_0 = self.conv0_2(attn_0)
#
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#
#         attn_2 = self.conv2_1(attn)
#         attn_2 = self.conv2_2(attn_0)
#         attn = attn + attn_0 + attn_1 + attn_2
#
#         attn = self.conv3(attn)
#
#         return attn * u



class ChannelAttention(nn.Module):
    # Channel-attention module
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


# class ChannelAttention(nn.Module):
#     # Channel-attention module
#     def __init__(self, channels: int) -> None:
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class MSAM(nn.Module):
    # Multi-Scale Attention Module
    def __init__(self, c1, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.multi_scale_conv = MultiScaleConv(c1, c1, kernel_sizes)
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.multi_scale_conv(x)
        x = self.channel_attention(x)
        return self.spatial_attention(x)

# class MSAM(nn.Module):
#     # Multi-Scale Attention Module
#     def __init__(self, c1, kernel_sizes=7):
#         super().__init__()
#         self.channel_attention = MSCAAttention(c1)
#         self.spatial_attention = SpatialAttention(kernel_sizes)
#
#     def forward(self, x):
#
#         return self.spatial_attention(self.channel_attention(x))