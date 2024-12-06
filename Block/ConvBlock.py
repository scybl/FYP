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
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, padding_mode=padding_mode, bias=bias),

            # 2. BatchNorm2d:
            #   - 对卷积后的输出进行批量归一化处理
            #   - 通过标准化每个小批次的输出，帮助网络更快收敛，同时减轻梯度消失和梯度爆炸问题
            nn.BatchNorm2d(out_channels),


            # 3. LeakyReLU:
            #   - 使用Leaky ReLU作为激活函数，默认负斜率为0.01，用于避免死神经元现象
            #   - 对小于0的输入乘以负斜率，使得负值部分不会完全为0
            nn.LeakyReLU(),  # 第一个卷积

            # 4. Dropout2d:
            #   - 以0.3的概率随机丢弃一些通道，以防止过拟合
            #   - 仅在训练过程中生效，在推理过程中关闭
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, x):
        return self.layer(x)
