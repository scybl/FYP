from torch import nn

from model.FYPNet.FYPNetBlock import CCBlock, DAG, PHAM, UCB
from model.FYPNet.ResNet import resnet34, resnet50, resnet101, resnet152


class Input_project(nn.Module):

    def __init__(self, in_channel):
        super(Input_project, self).__init__()
        self.preprocess = nn.Sequential(
            CCBlock(in_channel, 3)
        )
        pass

    def forward(self, x):
        return self.preprocess(x)


class Encoder(nn.Module):
    def __init__(self, encoder_mode='original', pretrain=True):
        super(Encoder, self).__init__()
        if encoder_mode == 'res34':
            self.backbone = resnet34(pretrained=pretrain)
        elif encoder_mode == 'res50':
            self.backbone = resnet50(pretrained=pretrain)
        elif encoder_mode == 'res101':
            self.backbone = resnet101(pretrained=pretrain)
        elif encoder_mode == 'res152':
            self.backbone = resnet152(pretrained=pretrain)
        elif encoder_mode == 'original':
            self.backbone = nn.Sequential()
        else:
            print('无效编码器')
            self.backbone = resnet50(pretrained=pretrain)

    def forward(self, x):
        return self.backbone(x)


class Decoder(nn.Module):
    def __init__(self, supervisor):
        super(Decoder, self).__init__()
        self.supervisor = supervisor
        self.dag1 = DAG(64)
        self.dag2 = DAG(128)
        self.dag3 = DAG(256)

        self.pham1 = PHAM(512)
        self.ucb1 = UCB(512, 256)

        self.pham2 = PHAM(256)
        self.ucb2 = UCB(256, 128)

        self.pham3 = PHAM(128)
        self.ucb3 = UCB(128, 64)

        self.pham4 = PHAM(64)

    def forward(self, R1, R2, R3, R4):
        out1 = self.pham1(R4)

        U1 = self.ucb1(out1)
        Dag1 = self.dag3(R3, U1)

        out2 = self.pham2(Dag1 + U1)

        U2 = self.ucb2(out2)
        Dag2 = self.dag2(R2, U2)

        out3 = self.pham3(Dag2 + U2)

        U3 = self.ucb3(out3)
        Dag3 = self.dag1(R1, U3)

        out4 = self.pham4(Dag3 + U3)

        if self.supervisor:
            return [out1, out2, out3, out4]
        else:
            return out4


class Output_project(nn.Module):
    """
    This class is used to handle the channel projection of the model's generated output.
    """

    def __init__(self, out_channel, supervisor):
        super(Output_project, self).__init__()
        self.supervisor = supervisor
        self.out_channel = out_channel
        self.out_project = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        return self.out_project(x)


class BNet_Res34(nn.Module):
    """
    according to the channel num to do
    """

    def __init__(self, in_channel, num_classes, encoder_mode, pre_train=True, deep_supervisor=False):
        super(BNet_Res34, self).__init__()
        self.input_project = Input_project(in_channel)

        self.encoder = Encoder(encoder_mode=encoder_mode, pretrain=pre_train)
        self.decoder = Decoder(deep_supervisor)

        self.out_project = Output_project(num_classes, deep_supervisor)

    def forward(self, _x):
        input = self.input_project(_x)

        R1, R2, R3, R4 = self.encoder(input)

        # print("中间layer的输出层级")
        # print(R1.shape)
        # print(R2.shape)
        # print(R3.shape)
        # print(R4.shape)

        out = self.decoder(R1, R2, R3, R4)
        out = self.out_project(out)

        return out
