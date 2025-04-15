from torch import nn
from model_defination.AAA_BNet.BNetBlock import CCBlock, DAG, PHAM, UCB, DownSample

class PHAMNet(nn.Module):
    """
    according to the channel num to do
    """

    def __init__(self, num_classes, in_channel, deep_supervisor=False):
        super(PHAMNet, self).__init__()
        self.supervisor = deep_supervisor

        self.cc1 = CCBlock(in_channel, 64)  # 转64通道

        self.down1 = DownSample(64)
        self.cc2 = CCBlock(64, 128)

        self.down2 = DownSample(128)
        self.cc3 = CCBlock(128, 256)

        self.d3 = DownSample(256)
        self.cc4 = CCBlock(256, 512)  # 512*32*32


        self.pham1 = CCBlock(512,512)
        self.ucb1 = UCB(512, 256)  # 同时缩减通道维度并扩张空间维度, (256,64,64)

        self.pham2 = CCBlock(256,256)
        self.ucb2 = UCB(256, 128)

        self.pham3 = CCBlock(128,128)
        self.ucb3 = UCB(128, 64)

        self.pham4 = CCBlock(64,64)

        self.convFinal1 = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1),
            # nn.BatchNorm2d(num_classes)
        )

    def forward(self, _x):
        R1 = self.cc1(_x)
        R2 = self.cc2(self.down1(R1))
        R3 = self.cc3(self.down2(R2))
        R4 = self.cc4(self.d3(R3))

        out1 = self.pham1(R4)

        U1 = self.ucb1(out1)

        out2 = self.pham2(U1)

        U2 = self.ucb2(out2)

        out3 = self.pham3(U2)

        U3 = self.ucb3(out3)

        out4 = self.convFinal1(self.pham4(U3))

        if self.supervisor:
            return [out1, out2, out3, out4]
        else:
            return out4

