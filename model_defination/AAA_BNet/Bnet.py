from torch import nn

from model_defination.AAA_BNet.BNetBlock import CCBlock, DAG, PHAM, UCB, DownSample

"""
这个是所有的B-Net架构的内容，我将所有封装的模块都写在这里，方便后续更改
"""

class BNet(nn.Module):
    """
    according to the channel num to do
    """

    def __init__(self, in_channel, num_classes, deep_supervisor = False):
        super(BNet, self).__init__()
        self.supervisor = deep_supervisor

        self.cc1 = CCBlock(in_channel, 64) # 转64通道

        self.down1 = DownSample(64)
        self.cc2 = CCBlock(64, 128)

        self.down2 = DownSample(128)
        self.cc3 = CCBlock(128, 256)

        self.d3 = DownSample(256)
        self.cc4 = CCBlock(256, 512)  # 512*32*32

        self.cc1 = CCBlock(in_channel, 3)


        self.dag1 = DAG(64)
        self.dag2 = DAG(128)
        self.dag3 = DAG(256)

        self.pham1 = PHAM(512)
        self.ucb1 = UCB(512, 256)  # 同时缩减通道维度并扩张空间维度, (256,64,64)

        self.pham2 = PHAM(256)
        self.ucb2 = UCB(256, 128)

        self.pham3 = PHAM(128)
        self.ucb3 = UCB(128, 64)

        self.pham4 = PHAM(64)

        self.convFinal1 = nn.Conv2d(64, num_classes, 1)


    def forward(self, _x):
        R1 = self.cc1(_x)
        R2 = self.cc2(self.down1(R1))
        R3 = self.cc3(self.down2(R2))
        R4 = self.cc4(self.d3(R3))

        R1, R2, R3, R4 = self.encoder(input)

        out1 = self.pham1(R4)

        U1 = self.ucb1(out1)
        Dag1 = self.dag3(R3, U1)

        out2 = self.pham2(Dag1 + U1)

        U2 = self.ucb2(out2)
        Dag2 = self.dag2(R2, U2)

        out3 = self.pham3(Dag2 + U2)

        U3 = self.ucb3(out3)
        Dag3 = self.dag1(R1, U3)

        out4 = self.convFinal1(self.pham4(Dag3 + U3))

        if self.supervisor:
            return [out1, out2, out3, out4]
        else:
            return out4
