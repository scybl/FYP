import torch
import torch.nn as nn
import torch.nn.functional as F


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
