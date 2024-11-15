from torch import nn


class SeBlock(nn.Module):
    """
    se通道注意路模块，通过计算每个通道的权重值，来得到每个通道的价值，进一步拟合结果
    """

    def __init__(self, mode, channels, ratio):
        super(SeBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if mode == "max":
            self.global_pool = self.max_pool
        elif mode == "avg":
            self.global_pool = self.avg_pool
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.global_pool(x).view(b, c)  #
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x * v
