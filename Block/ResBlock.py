from torch import nn
from Block.ConvBlock import ConvBlock2


class ResidualBlock(nn.Module):
    """
    残差块
    如果输入和输出的通道数不一致，则调整输入的通道数以确保可以相加。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                 bias=False, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()

        # 定义卷积块
        self.conv_block = ConvBlock2(in_channels, out_channels, kernel_size, stride, padding, padding_mode, bias,
                                    dropout_rate)

        # 如果输入通道数与输出通道数不匹配，通过 1x1 卷积进行调整
        self.adjust_channels = None
        if in_channels != out_channels:
            self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # 通过卷积块处理输入
        out = self.conv_block(x)

        # 如果通道数不一致，通过 1x1 卷积调整输入的通道数
        residual = self.adjust_channels(x) if self.adjust_channels else x

        # 将残差连接加入卷积块的输出

        return out + residual
