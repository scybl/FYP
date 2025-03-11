from torch import nn
import torch


class SpatialAttention(nn.Module):
    """
    Spatial attention module
    """

    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.convl = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.convl(out)
        out = self.sigmoid(out)
        return out


class ChannelAttention(nn.Module):
    """
    Channel attention module
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)  # this is the full-connection layer
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)

        return out


class CbamBlock(nn.Module):
    """
    cbam注意力机制结合了**通道注意力**和**空间注意力机制**，
    包含了cam通道注意力模块模块和sam空间注意力模块，
    分别对通道和空间上的注意力进行特征提取融合
    """

    def __init__(self, in_channels, ratio=4, kernel_size=7):
        super(CbamBlock, self).__init__()

        self.channel_attention = ChannelAttention(in_channels, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x
