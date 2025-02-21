import torch
from torch import nn
from torch.nn import functional as F

from model_defination.AAA_BNet.DAG import DAG
from model_defination.AAA_BNet.PHAM import PHAM
from model_defination.AAA_BNet.UCB import UCB


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
    Down sample will cut the pixel in half using Max Pooling
    """

    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化
            nn.BatchNorm2d(channel),  # 保持批量归一化
            nn.LeakyReLU()  # 保持激活函数
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
        self.cc1 = BizareBlock(3, 64)
        self.dag1 = DAG(64)
        self.down1 = DownSample(64)

        self.cc2 = BizareBlock(64, 128)
        self.dag2 = DAG(128)
        self.down2 = DownSample(128)

        self.cc3 = BizareBlock(128, 256)
        self.dag3 = DAG(256)
        self.d3 = DownSample(256)

        self.cc4 = BizareBlock(256, 512)  # 512*32*32

        self.pham1 = PHAM(512)
        self.ucb1 = UCB(512, 256)  # 同时缩减通道维度并扩张空间维度, (256,64,64)

        self.pham2 = PHAM(256)
        self.ucb2 = UCB(256, 128)

        self.pham3 = PHAM(128)
        self.ucb3 = UCB(128, 64)

        self.pham4 = PHAM(64)

        self.convFinal1 = nn.Conv2d(64, num_classes, 1)

    def forward(self, _x):
        R1 = self.cc1(_x)
        R2 = self.cc2(self.down1(R1))
        R3 = self.cc3(self.down2(R2))
        R4 = self.cc4(self.d3(R3))

        U1 = self.ucb1(self.pham1(R4))

        Dag1 = self.dag3(R3, U1)
        plus1_out = Dag1 + U1

        U2 = self.ucb2(self.pham2(plus1_out))
        Dag2 = self.dag2(R2, U2)
        plus2_out = Dag2 + U2

        U3 = self.ucb3(self.pham3(plus2_out))
        Dag3 = self.dag1(R1, U3)
        plus3_out = Dag3 + U3

        pham4_out = self.pham4(plus3_out)

        return self.convFinal1(pham4_out)
