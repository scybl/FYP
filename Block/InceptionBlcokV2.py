import torch
from torch import nn

from Block.ConvBlock import ConvBlock


class InceptionBlockV2(nn.Module):
    """
    this is inception V2 block,
    use two 3*3 conv core to replace the 5*5 core
    """

    def __init__(self, channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        """
        Inception Block with dimension reduction.
        Args:
            out_1x1: 25%

            red_3x3: 25%
            out_3x3: 50%

            red_5x5: 6.25%
            out_5x5: 12.5%

            out_pool: 12.5%
        """
        super(InceptionBlockV2, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(channels, ch1x1, 1, 1, 0)
        )

        self.branch2 = nn.Sequential(
            ConvBlock(channels, ch3x3red, 1, 1, 0),
            ConvBlock(ch3x3red, ch3x3, 3, 1, 1)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(channels, ch5x5red, 1, 1, 0),
            ConvBlock(ch5x5red, ch5x5, 3, 1, 1),
            ConvBlock(ch5x5, ch5x5, 3, 1, 1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            ConvBlock(channels, pool_proj, 1, 1, 0)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], 1)
