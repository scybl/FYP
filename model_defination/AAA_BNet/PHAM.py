import torch
import torch.nn as nn

from model_defination.AAA_BNet.ECAB import ECAB
from model_defination.AAA_BNet.ESAB import ESAB


class PHAM(nn.Module):
    def __init__(self, channels, dropout=0.3):
        super(PHAM, self).__init__()

        # ECAB 和 ESAB 分支
        self.ecab = ECAB(channels)  # ECAB模块，需提供实际实现
        self.esab = ESAB(channels)  # ESAB模块，需提供实际实现

        # 1x1 卷积分支
        self.conv1xLayer = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=dropout),  # Dropout layer
        )

        # 3x3 卷积分支
        self.conv3xLayer = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=dropout),  # Dropout layer
        )

        # 5x5 卷积分支（可以用两次3x3卷积替代）
        self.conv5xLayer = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=dropout),  # Dropout layer
        )

        # **新增：可学习参数**
        self.weights = nn.Parameter(torch.ones(3))  # 初始化三个分支的权重

    @staticmethod
    def channel_shuffle(x, groups):
        """
        通道打乱操作，用于重排列通道。
        """
        batch_size, num_channels, height, width = x.size()
        assert num_channels % groups == 0, "通道数必须可以被组数整除"
        group_channels = num_channels // groups

        # 调整维度为 (batch_size, groups, group_channels, height, width)
        x = x.view(batch_size, groups, group_channels, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # 转置通道和组的维度

        # 恢复形状为 (batch_size, num_channels, height, width)
        x = x.view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        # ECAB 和 ESAB 输出
        ecab_out = self.ecab(x)
        esab_out = self.esab(x)

        # 按通道拼接 (C, H, W) -> (C*2, H, W)
        combined = torch.cat([ecab_out, esab_out], dim=1)

        # 三个卷积分支输出
        out1 = self.conv1xLayer(combined)
        out2 = self.conv3xLayer(combined)
        out3 = self.conv5xLayer(combined)

        # **使用 softmax 归一化可学习参数**
        weight = torch.softmax(self.weights, dim=0)  # 归一化，确保总和为 1

        # **动态调整各分支比重**
        out = weight[0] * out1 + weight[1] * out2 + weight[2] * out3

        # 通道打乱
        out = self.channel_shuffle(out, groups=2)
        return out
