import torch
import torch.nn as nn


class ECAB(nn.Module):
    def __init__(self, channels, reduction_ratio=4, dropout_rate=0.3):
        super(ECAB, self).__init__()
        reduced_channels = channels // reduction_ratio

        # AMP Path (Global Average Pooling)
        self.AMP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling (C, H, W) -> (C, 1, 1)
            nn.Conv2d(channels, reduced_channels, 1),  # Conv 1x1: (C, 1, 1) -> (C/r, 1, 1)
            nn.ReLU(inplace=True),  # ReLU Activation
            nn.Dropout(p=dropout_rate),  # Dropout layer
            nn.Conv2d(reduced_channels, channels, 1)  # Conv 1x1: (C/r, 1, 1) -> (C, 1, 1)
        )

        # APP Path (Global Max Pooling)
        self.APP = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # Global Max Pooling (C, H, W) -> (C, 1, 1)
            nn.Conv2d(channels, reduced_channels, 1),  # Conv 1x1: (C, 1, 1) -> (C/r, 1, 1)
            nn.ReLU(inplace=True),  # ReLU Activation
            nn.Dropout(p=dropout_rate),  # Dropout layer
            nn.Conv2d(reduced_channels, channels, 1)  # Conv 1x1: (C/r, 1, 1) -> (C, 1, 1)
        )

        # Spatial H * W Convolution Path
        self.layer3 = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1),  # Conv 1x1: (C, H, W) -> (C/r, H, W)
            nn.ReLU(inplace=True),  # ReLU Activation
            nn.Dropout(p=dropout_rate),  # Dropout layer
            nn.Conv2d(reduced_channels, channels, 1)  # Conv 1x1: (C/r, H, W) -> (C, H, W)
        )

        # Sigmoid activation for attention scaling
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute AMP attention
        amp_out = self.AMP(x)

        # Compute APP attention
        app_out = self.APP(x)

        # Compute spatial H * W convolution attention
        hw_out = self.layer3(x)

        # Combine all three paths
        combined = amp_out + app_out + hw_out

        # Apply sigmoid to compute the scaling factor
        scale = self.sigmoid(combined)

        # Scale the input by the attention map
        out = x * scale
        return out
