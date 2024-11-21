import torch
import torch.nn as nn
import torch.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block: Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """Encoder: Down-sampling using ConvBlocks and MaxPooling"""

    def __init__(self, in_channels, feature_channels):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for channels in feature_channels:
            self.layers.append(ConvBlock(in_channels, channels))
            in_channels = channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
            x = self.pool(x)
        return features, x


class Decoder(nn.Module):
    """Decoder: Up-sampling and merging with skip connections"""

    def __init__(self, feature_channels):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(feature_channels) - 1, 0, -1):
            self.layers.append(nn.ConvTranspose2d(
                feature_channels[i], feature_channels[i - 1],
                kernel_size=2, stride=2))
            self.layers.append(ConvBlock(
                feature_channels[i], feature_channels[i - 1]))

    def forward(self, x, encoder_features):
        for i in range(len(self.layers) // 2):
            x = self.layers[2 * i](x)
            skip = encoder_features[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = self.layers[2 * i + 1](x)
        return x


class UNet(nn.Module):
    """UNet framework"""

    def __init__(self, in_channels=3, out_channels=1, feature_channels=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, feature_channels)
        self.decoder = Decoder(feature_channels)
        self.final_conv = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_features, bottleneck = self.encoder(x)
        x = self.decoder(bottleneck, encoder_features)
        return self.final_conv(x)


# Example: Define a UNet model with flexible configurations
if __name__ == "__main__":
    # Define model
    model = UNet(in_channels=3, out_channels=1, feature_channels=[64, 128, 256, 512])
    print(model)

    # Test the model with dummy input
    input_tensor = torch.randn(1, 3, 256, 256)  # Batch size=1, 3 channels, 256x256
    output = model(input_tensor)
    print("Output shape:", output.shape)