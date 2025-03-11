import torch
from torch import nn
from torch.nn import functional as F

"""
这个是所有的B-Net架构的内容，我将所有封装的模块都写在这里，方便后续更改
"""


class DAG(nn.Module):
    def __init__(self, channels, dilation_rate=2, dropout_rate=0.0):
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


class ECAB(nn.Module):
    def __init__(self, channels, deep_supervisor=True, reduction_ratio=4, dropout_rate=0.0):
        super(ECAB, self).__init__()
        reduced_channels = channels // reduction_ratio
        self.supervisor = deep_supervisor

        # AMP 路径（全局平均池化） 使用线性层
        self.AMP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化: (B, C, H, W) -> (B, C, 1, 1)
            nn.Flatten(),  # 展平: (B, C, 1, 1) -> (B, C)
            nn.Linear(channels, reduced_channels),  # 线性层: 降维 (B, C) -> (B, C/r)
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Dropout(p=dropout_rate),  # Dropout 层
            nn.Linear(reduced_channels, channels),  # 线性层: 升维 (B, C/r) -> (B, C)
            nn.Unflatten(1, (channels, 1, 1))  # 恢复形状: (B, C) -> (B, C, 1, 1)
        )

        # APP 路径（全局最大池化） 使用线性层
        self.APP = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # 全局最大池化: (B, C, H, W) -> (B, C, 1, 1)
            nn.Flatten(),  # 展平: (B, C, 1, 1) -> (B, C)
            nn.Linear(channels, reduced_channels),  # 线性层: 降维 (B, C) -> (B, C/r)
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Dropout(p=dropout_rate),  # Dropout 层
            nn.Linear(reduced_channels, channels),  # 线性层: 升维 (B, C/r) -> (B, C)
            nn.Unflatten(1, (channels, 1, 1))  # 恢复形状: (B, C) -> (B, C, 1, 1)
        )

        # 空间 H * W 卷积路径
        self.layer3 = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1),  # 1x1 卷积: (C, H, W) -> (C/r, H, W)
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Dropout(p=dropout_rate),  # Dropout 层
            nn.Conv2d(reduced_channels, channels, 1)  # 1x1 卷积: (C/r, H, W) -> (C, H, W)
        )

        # **新增：可学习参数**
        self.weights = nn.Parameter(torch.ones(3))  # 初始化三个路径的权重

        # Sigmoid 激活函数用于计算注意力缩放因子
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        amp_out = self.AMP(x)  # 计算 AMP 注意力
        app_out = self.APP(x)  # 计算 APP 注意力
        hw_out = self.layer3(x)  # 计算空间 H * W 卷积注意力

        # **使用 softmax 归一化可学习参数**
        weight = torch.softmax(self.weights, dim=0)  # 归一化权重，确保总和为 1

        # **动态调整路径比重**
        combined = weight[0] * amp_out + weight[1] * app_out + weight[2] * hw_out

        # 通过 Sigmoid 计算缩放因子
        scale = self.sigmoid(combined)

        # 通过注意力映射对输入进行缩放
        out = x * scale
        return out


class ESAB(nn.Module):
    def __init__(self, channels):
        super(ESAB, self).__init__()

        # Channel-wise pooling
        self.channel_max_pool = nn.AdaptiveMaxPool2d((None, None))  # Max over channels
        self.channel_avg_pool = nn.AdaptiveAvgPool2d((None, None))  # Avg over channels

        # 1x1 Conv for the original input
        self.conv1x1 = nn.Conv2d(channels, 1, kernel_size=1)

        # Dilated convolution
        self.dilated_conv = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=2, dilation=2)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = torch.max(x, dim=1, keepdim=True)[0]  # Max along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel Avg Pooling
        conv_out = self.conv1x1(x)  # 1x1 Conv

        # Concatenate outputs (3 channels: Max, Avg, 1x1 Conv output)
        combined = torch.cat([max_out, avg_out, conv_out], dim=1)

        dilated_out = self.dilated_conv(combined)  # Dilated convolution
        scale = self.sigmoid(dilated_out)  # Sigmoid scaling

        out = x * scale  # Scale input feature map
        return out


class PHAM(nn.Module):
    def __init__(self, channels, dropout=0.0):
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
        通道打乱操作，用于重排列通道。使用静态微加速
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


class UCB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UCB, self).__init__()

        # 上采样（2倍，双线性插值）
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # ConvTranspose

        # 深度可分离卷积 (Depthwise Separable Convolution, DWC)
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Batch Normalization 和 ReLU
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # GeLU, LeakRelU, ReLU6()

    def forward(self, x):
        # 执行上采样
        x = self.upsample(x)

        # 深度可分离卷积
        x = self.dw_conv(x)
        x = self.pointwise_conv(x)

        # BN 和 ReLU
        x = self.bn(x)
        x = self.relu(x)

        return x


class CCBlock(nn.Module):
    """
    连续卷积模块
    """

    def __init__(self, in_channel, out_channel, drop_rate=0.0):
        super(CCBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(drop_rate),
            nn.LeakyReLU(),
            # dilation mean the
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False, dilation=1),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(drop_rate),
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
