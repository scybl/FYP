import torch.nn as nn

from Block.ConvBlock import ConvBlock


class DownSample(nn.Module):
    def __init__(self, in_channels):
        super(DownSample, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)


class FCN8s(nn.Module):
    def __init__(self, num_class=3):
        super(FCN8s, self).__init__()
        self.n_class = num_class

        # VGG Backbone
        self.conv1 = ConvBlock(3, 64, 3, 1, 1)
        self.pool1 = DownSample(64)  # [N,3,256,256]->[N,64,128,128]

        self.conv2 = ConvBlock(64, 128, 3, 1, 1)
        self.pool2 = DownSample(128)  # [N,64,128,128]->[N,128,64,64]

        self.conv3 = ConvBlock(128, 256, 3, 1, 1)
        self.pool3 = DownSample(256)  # [N,128,64,64]->[N,256,32,32]

        self.conv4 = ConvBlock(256, 512, 3, 1, 1)
        self.pool4 = DownSample(512)  # [N,256,32,32]->[N,512,16,16]

        self.conv5 = ConvBlock(512, 512, 3, 1, 1)
        self.pool5 = DownSample(512)  # [N,512,16,16]->[N,512,16,16]

        # Fully Connected Layers (as convolutions)
        self.conv6 = nn.Conv2d(512, 4096, kernel_size=8)
        nn.init.xavier_uniform_(self.conv6.weight)  # [N,512,16,16]->[N,4096,16,16]
        # xavier_uniform_ 是一种权重初始化方法，用于神经网络中的权重矩阵初始化
        # 它的主要目的是使得每一层的输入和输出的方差尽可能相等，从而帮助网络在训练时更稳定地收敛。

        self.bn6 = nn.BatchNorm2d(4096)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        nn.init.xavier_uniform_(self.conv7.weight)
        self.bn7 = nn.BatchNorm2d(4096)
        self.relu7 = nn.ReLU(inplace=True)

        # Classifier
        self.score_fr = nn.Conv2d(4096, num_class, kernel_size=1)
        nn.init.xavier_uniform_(self.score_fr.weight)

        # Upsampling Layers
        self.upscore2 = nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2)
        nn.init.xavier_uniform_(self.upscore2.weight)

        self.score_pool4 = nn.Conv2d(512, num_class, kernel_size=1)
        nn.init.xavier_uniform_(self.score_pool4.weight)

        self.upscore_pool4 = nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2)
        nn.init.xavier_uniform_(self.upscore_pool4.weight)

        self.score_pool3 = nn.Conv2d(256, num_class, kernel_size=1)
        nn.init.xavier_uniform_(self.score_pool3.weight)

        self.upscore8 = nn.ConvTranspose2d(num_class, num_class, kernel_size=16, stride=8)
        nn.init.xavier_uniform_(self.upscore8.weight)

    def forward(self, x):
        # Feature extraction through encoder
        x1 = self.conv1(x)
        p1 = self.pool1(x1)
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)
        x3 = self.conv3(p2)
        p3 = self.pool3(x3)
        x4 = self.conv4(p3)
        p4 = self.pool4(x4)
        x5 = self.conv5(p4)
        p5 = self.pool5(x5)

        # Fully connected convolutional layers
        x6 = self.relu6(self.bn6(self.conv6(p5)))
        x7 = self.relu7(self.bn7(self.conv7(x6)))

        # Upsampling
        sf = self.score_fr(x7)
        u2 = self.upscore2(sf)

        s4 = self.score_pool4(p4)
        print(s4.size())
        print(u2.size())
        f4 = s4 + u2
        u4 = self.upscore_pool4(f4)

        s3 = self.score_pool3(p3)
        f3 = s3 + u4
        out = self.upscore8(f3)

        return out
