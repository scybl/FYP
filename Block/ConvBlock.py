from torch import nn


class ConvBlock(nn.Module):
    """
    卷积块
    ReLU : f(x) = max(0, x)
    Sigmoid : f(x) = 1 / (1 + exp(-x))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_mode="reflect", bias=False,
                 dropout_rate=0.3):
        # 调用父类的初始化方法
        super(ConvBlock, self).__init__()

        # 定义卷积块 (ConvBlock)，其中包含以下层：
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, padding_mode=padding_mode, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, x):
        return self.layer(x)
