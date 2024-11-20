import torch
import torch.nn as nn
import torch.utils.data

from Block.ConvBlock import ConvBlock
from Block.Sample import UpSample

# TODO: need to upSample to the size * size PIC
class VGG16(nn.Module):
    def __init__(self, num_classes=3):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            ConvBlock(3, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),

            # Block 2
            UpSample(64),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),

            # Block 3
            UpSample(128),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),

            # Block 4
            UpSample(256),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),

            # Block 5
            UpSample(512),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),

            # final block
            UpSample(512),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),  # this size must be attention
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展开特征图
        x = self.classifier(x)

        return x
