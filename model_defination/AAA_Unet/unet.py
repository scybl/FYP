from torch import nn

from Block.ConvBlock import ConvBlock
from Block.Sample import DownSample, UpSample


class UNetBase(nn.Module):
    """
    this is the base U-net model, It is copy from the paper nearly total.
    the changed point is change the max pool to the 2-size conv core
    """

    def __init__(self, in_channel, class_num=1):
        super(UNetBase, self).__init__()
        # ------开始下采样
        self.layer1 = nn.Sequential(
            ConvBlock(in_channel, 64, 3, 1, 1, "reflect", False, 0.3),
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

        self.layer5 = nn.Sequential(
            ConvBlock(512, 1024, 3, 1, 1, "reflect", False, 0.3),
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False, dropout_rate=0.3)
        )

        # -------开始上采样
        self.up1 = UpSample(1024)
        self.layer6 = nn.Sequential(
            ConvBlock(1024, 512, 3, 1, 1, "reflect", False, 0.3),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False, dropout_rate=0.3)
        )

        self.up2 = UpSample(512)
        self.layer7 = nn.Sequential(
            ConvBlock(512, 256, 3, 1, 1, "reflect", False, 0.3),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False, dropout_rate=0.3)
        )

        self.up3 = UpSample(256)
        self.layer8 = nn.Sequential(
            ConvBlock(256, 128, 3, 1, 1, "reflect", False, 0.3),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False, dropout_rate=0.3)
        )

        self.up4 = UpSample(128)
        self.layer9 = nn.Sequential(
            ConvBlock(128, 64, 3, 1, 1, "reflect", False, 0.3),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False, dropout_rate=0.3)
        )

        self.out = nn.Sequential(
            nn.Conv2d(64, class_num, 3, 1, 1)
        )

    def forward(self, x):
        # encode
        L1 = self.layer1(x)
        L2 = self.layer2(self.d1(L1))
        L3 = self.layer3(self.d2(L2))
        L4 = self.layer4(self.d3(L3))
        L5 = self.layer5(self.d4(L4))

        # decode
        O1 = self.layer6(self.up1(L5, L4))
        O2 = self.layer7(self.up2(O1, L3))
        O3 = self.layer8(self.up3(O2, L2))
        O4 = self.layer9(self.up4(O3, L1))

        return self.out(O4)  # 所谓图像分割就是生成一个预期的图片，这个图片大小与输入的图片大小相等
