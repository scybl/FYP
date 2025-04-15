import torch
from torch import nn
from Block.ConvBlock import ConvBlock2


class InceptionBlockV3(nn.Module):
    """
    This is the Inception V3 block.
    Key differences:
    - Replace 5x5 convolutions with two consecutive 3x3 convolutions.
    - Use asymmetric factorization for 3x3 (split into 1x3 and 3x1).
    """

    def __init__(self, channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        """
        Inception Block with advanced techniques from Inception V3.
        Args:
            channels: Input channel size.
            ch1x1: Number of filters in the 1x1 convolution branch.
            ch3x3red: Number of filters in the 1x1 reduction for the 3x3 branch.
            ch3x3: Number of filters in the 3x3 branch after reduction.
            ch5x5red: Number of filters in the 1x1 reduction for the factorized 5x5 branch.
            ch5x5: Number of filters in the two 3x3 convolutions (replacing 5x5).
            pool_proj: Number of filters in the pooling branch.
        """
        super(InceptionBlockV3, self).__init__()

        # Branch 1: 1x1 Convolution
        self.branch1 = nn.Sequential(
            ConvBlock2(channels, ch1x1, 1, 1, 0)
        )

        # Branch 2: 1x1 Reduction + 3x3 Factorized into 1x3 and 3x1
        self.branch2 = nn.Sequential(
            ConvBlock2(channels, ch3x3red, 1, 1, 0),
            ConvBlock2(ch3x3red, ch3x3, (1, 3), 1, (0, 1)),
            ConvBlock2(ch3x3, ch3x3, (3, 1), 1, (1, 0))
        )

        # Branch 3: 1x1 Reduction + Two 3x3 Convolutions (Replacing 5x5)
        self.branch3 = nn.Sequential(
            ConvBlock2(channels, ch5x5red, 1, 1, 0),
            ConvBlock2(ch5x5red, ch5x5, 3, 1, 1),
            ConvBlock2(ch5x5, ch5x5, 3, 1, 1)
        )

        # Branch 4: Pooling + 1x1 Convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock2(channels, pool_proj, 1, 1, 0)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], 1)
