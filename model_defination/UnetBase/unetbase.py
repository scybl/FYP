from torch import nn

from Block.ConvBlock import ConvBlock
from Block.Sample import DownSample, UpSample


class UNetBase(nn.Module):
    """
    this is the base U-net model, It is copy from the paper nearly total.
    the changed point is change the max pool to the 2-size conv core
    """

    def __init__(self, class_num=1):
        super(UNetBase, self).__init__()
        # ------开始下采样
        self.layer1 = nn.Sequential(
            ConvBlock(3, 64, 3, 1, 1, "reflect", False, 0.3),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False, dropout_rate=0.3)
        )
        self.d1 = DownSample(64)

        self.layer2 = nn.Sequential(
            ConvBlock(64, 128, 3, 1, 1, "reflect", False, 0.3),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False, dropout_rate=0.3)
        )
        self.d2 = DownSample(128)

        self.layer3 = nn.Sequential(
            ConvBlock(128, 256, 3, 1, 1, "reflect", False, 0.3),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False, dropout_rate=0.3)
        )
        self.d3 = DownSample(256)

        self.layer4 = nn.Sequential(
            ConvBlock(256, 512, 3, 1, 1, "reflect", False, 0.3),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False, dropout_rate=0.3)
        )
        self.d4 = DownSample(512)

        self.conv5_0 = ConvBlock(512, 1024, 3, 1, 1, "reflect", False, 0.3)
        self.conv5 = ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                               bias=False, dropout_rate=0.3)

        # -------开始下采样
        self.up1 = UpSample(1024)
        self.conv6_0 = ConvBlock(1024, 512, 3, 1, 1, "reflect", False, 0.3)
        self.conv6 = ConvBlock(512, 512, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                               bias=False, dropout_rate=0.3)

        self.up2 = UpSample(512)
        self.conv7_0 = ConvBlock(512, 256, 3, 1, 1, "reflect", False, 0.3)
        self.conv7 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                               bias=False, dropout_rate=0.3)

        self.up3 = UpSample(256)
        self.conv8_0 = ConvBlock(256, 128, 3, 1, 1, "reflect", False, 0.3)
        self.conv8 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                               bias=False, dropout_rate=0.3)

        self.up4 = UpSample(128)
        self.conv9_0 = ConvBlock(128, 64, 3, 1, 1, "reflect", False, 0.3)
        self.conv9 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                               bias=False, dropout_rate=0.3)

        self.out = nn.Conv2d(64, class_num, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()  # 使用sigmoid进行二分

    def forward(self, x):
        # encode
        R1 = self.conv1(self.conv1_0(x))
        R2 = self.conv2(self.conv2_0(self.d1(R1)))
        R3 = self.conv3(self.conv3_0(self.d2(R2)))
        R4 = self.conv4(self.conv4_0(self.d3(R3)))
        R5 = self.conv5(self.conv5_0(self.d4(R4)))

        # decode
        t1 = self.up1(R5, R4)
        O1 = self.conv6(self.conv6_0(t1))
        O2 = self.conv7(self.conv7_0(self.up2(O1, R3)))
        O3 = self.conv8(self.conv8_0(self.up3(O2, R2)))
        O4 = self.conv9(self.conv9_0(self.up4(O3, R1)))
        return self.out(O4)  # 所谓图像分割就是生成一个预期的图片，这个图片大小与输入的图片大小相等
