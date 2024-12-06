import torch
import torch.nn as nn


class DAG(nn.Module):
    def __init__(self, channels, dilation_rate=2, dropout_rate=0.3):
        super(DAG, self).__init__()

        # 第一个膨胀卷积分支
        self.dilated_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout_rate)
        )

        # 第二个膨胀卷积分支
        self.dilated_conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout_rate)
        )

        # 1x1 卷积用于降维
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout_rate)
        )

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

        # Sigmoid 用于生成注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # 第一个膨胀卷积分支
        branch1 = self.dilated_conv1(x1)

        # 第二个膨胀卷积分支
        branch2 = self.dilated_conv2(x2)

        # 两个分支结果相加
        combined = branch1 + branch2

        out = self.relu(combined)
        out = self.conv1x1(out)

        # Sigmoid 激活生成注意力权重
        attention_map = self.sigmoid(out)

        # 输入乘以注意力权重
        out = x2 * attention_map
        return out
