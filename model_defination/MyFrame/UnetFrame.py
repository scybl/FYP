import torch
from torch import nn
from torch.nn import functional as F

from Block.InceptionBlcokV2 import InceptionBlockV2
from Block.InceptionBlcokV3 import InceptionBlockV3


class BizareBlock(nn.Module):
    """
    This class is controlled by the in_channel and out_channel arguments, to control the number of channels
    """

    def __init__(self, in_channel, out_channel):
        super(BizareBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            # dilation mean the
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False, dilation=1),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, _x):
        return self.layer(_x)


class DownSample(nn.Module):
    """
    Down sample will cut the pixel in half
    """

    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, _x):
        return self.layer(_x)


class UpSample(nn.Module):
    """
    Up sample will double the pixel
    """

    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, _x, feature_map):
        up = F.interpolate(_x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


class BNet(nn.Module):
    """
    according to the channel num to do
    """

    def __init__(self, num_classes=1):
        super(BNet, self).__init__()
        self.c1 = BizareBlock(3, 64)
        self.d1 = DownSample(64)

        self.c2 = InceptionBlockV3(64, 32, 32, 64, 8, 16, 16)
        self.d2 = DownSample(128)

        self.c3 = InceptionBlockV3(128, 64, 64, 128, 16, 32, 32)
        self.d3 = DownSample(256)

        self.c4 = InceptionBlockV3(256, 128, 128, 256, 32, 64, 64)
        self.d4 = DownSample(512)

        self.c5 = InceptionBlockV3(512, 256, 256, 512, 64, 128, 128)
        self.u1 = UpSample(1024)

        self.c6 = BizareBlock(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = BizareBlock(512, 256)
        self.u3 = UpSample(256)
        self.c8 = BizareBlock(256, 128)
        self.u4 = UpSample(128)
        self.c9 = BizareBlock(128, 64)
        self.out = nn.Conv2d(64, num_classes, 3, 1, 1)

    def forward(self, _x):
        R1 = self.c1(_x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))

        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.out(O4)
