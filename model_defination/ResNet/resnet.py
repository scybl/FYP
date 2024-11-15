import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    # in_channel:输入block之前的通道数
    # channel:在block中间处理的时候的通道数（这个值是输出维度的1/4)
    # channel * block.expansion:输出的维度
"""


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, channel, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, channel, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)

        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn2 = nn.BatchNorm2d(channel)

        self.conv3 = nn.Conv2d(channel, channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


# 	BottleNeck：决定了每个残差块的结构，用于深层网络，帮助在网络中高效传递梯度。
# 	layer：指定每个层级的残差块数量，从而决定网络深度。
# 	num_classes：决定网络输出层的类别数，分类任务中是类别数量，分割任务中是分割的类别数量。
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, out_size=256):
        super(ResNet, self).__init__()
        self.out_size = out_size
        self.in_channel = 64

        # Stem layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Up-sampling layers
        self.upsample1 = nn.ConvTranspose2d(512 * block.expansion, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)



        # Final segmentation head
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion),
            )

        layers = [block(self.in_channel, channel, stride, downsample)]
        self.in_channel = channel * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem layer
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Up-sampling
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)

        # Segmentation head
        x = self.segmentation_head(x)


        # Resize output to match input size
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        return x


# 二分任务，默认为3,其他再改
def ResNet50(num_classes=3):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=3):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=3):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)

# if __name__ == '__main__':
#     input = torch.randn(50, 3, 224, 224)
#     resnet50 = ResNet50(1000)
#     out = resnet50(input)
#     print(out.shape)

# down code is the original code of resnet

# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=1000):
#         super().__init__()
#         # 定义输入模块的维度
#         self.in_channel = 64
#         # stem layer
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(False)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
#
#         # main layer
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#
#         # classifier, for segment is useless
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(512 * block.expansion, num_classes)
#         self.softmax = nn.Softmax(-1)
#
#     def forward(self, x):
#         # stem layer
#         _out = self.relu(self.bn1(self.conv1(x)))  # bs,112,112,64
#         _out = self.maxpool(_out)  # bs,56,56,64
#
#         # layers:
#         _out = self.layer1(_out)  # bs,56,56,64*4
#         _out = self.layer2(_out)  # bs,28,28,128*4
#         _out = self.layer3(_out)  # bs,14,14,256*4
#         _out = self.layer4(_out)  # bs,7,7,512*4
#
#         # classifier
#         _out = self.avgpool(_out)  # bs,1,1,512*4
#         _out = _out.reshape(_out.shape[0], -1)  # bs,512*4
#         _out = self.classifier(_out)  # bs,1000
#         _out = self.softmax(_out)
#
#         return _out
#
#     def _make_layer(self, block, channel, blocks, stride=1):
#         # downsample 主要用来处理H(x)=F(x)+x中F(x)和x的channel维度不匹配问题，即对残差结构的输入进行升维，在做残差相加的时候，必须保证残差的纬度与真正的输出维度（宽、高、以及深度）相同
#         # 比如步长！=1 或者 in_channel!=channel&self.expansion
#         downsample = None
#         if (stride != 1 or self.in_channel != channel * block.expansion):
#             self.downsample = nn.Conv2d(self.in_channel, channel * block.expansion, stride=stride, kernel_size=1,
#                                         bias=False)
#         # 第一个conv部分，可能需要downsample
#         layers = []
#         layers.append(block(self.in_channel, channel, downsample=self.downsample, stride=stride))
#         self.in_channel = channel * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channel, channel))
#         return nn.Sequential(*layers)
