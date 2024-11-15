from torch import nn


class ConvBlock(nn.Module):
    """
    卷积块
    ReLU : f(x) = max(0, x)
    Sigmoid : f(x) = 1 / (1 + exp(-x))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_mode="reflect", bias=False,
                 dropout_rate=0.3):
        # 调用父类的初始化方法
        super(ConvBlock, self).__init__()

        # 定义卷积块 (ConvBlock)，其中包含以下层：
        self.layer = nn.Sequential(
            # 1. Conv2d:
            #   - in_channels: 输入特征图的通道数（例如，对于RGB图像，in_channels = 3）
            #   - out_channels: 输出特征图的通道数，即卷积核的数量，这决定输出多少个特征图
            #   - kernel_size: 卷积核的大小为 3x3，表示卷积核将每次处理 3x3 区域的像素
            #   - stride=1: 步长为1，表示卷积核每次滑动1个像素，这使得特征图保持较高的分辨率
            #   - padding=1: 在输入图像的每条边缘填充1层像素，这样可以保持输出与输入的空间维度相同
            #   - padding_mode='reflect': 边界反射填充，即通过镜像反射边缘像素来填充图像的边缘, 默认填充0是没有任何特征的，使用反射填充相邻点的对称点，以保留特征加强特征提取能力
            #   - bias=False: 不使用偏置项，因为批量归一化层能够通过自身调整激活值，因此可以省略偏置项
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, padding_mode=padding_mode, bias=bias),

            # 2. BatchNorm2d:
            #   - 对卷积后的输出进行批量归一化处理
            #   - 通过标准化每个小批次的输出，帮助网络更快收敛，同时减轻梯度消失和梯度爆炸问题
            # disadvantage:
            # 1. only be useful for too much train sample
            # 2. for RNN and sequence have less effective
            # 3. add a new parameter to model, influence the effect
            nn.BatchNorm2d(out_channels),

            # 3. Dropout2d:
            #   - 以0.3的概率随机丢弃一些通道，以防止过拟合
            #   - 仅在训练过程中生效，在推理过程中关闭
            nn.Dropout2d(dropout_rate),

            # 4. LeakyReLU:
            #   - 使用Leaky ReLU作为激活函数，默认负斜率为0.01，用于避免死神经元现象
            #   - 对小于0的输入乘以负斜率，使得负值部分不会完全为0
            nn.LeakyReLU(),  # 第一个卷积
        )

    def forward(self, x):
        return self.layer(x)
