from model_defination.AAA_Unet.unet import UNetBase
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table


class UNetAnalyzer:
    """
    用于加载 UNet 模型，并计算其 FLOPs 和参数数量。

    参数:
    in_channels (int): 模型输入的通道数。
    device (torch.device): 设备，默认为cuda（如果可用）或cpu。
    """
    def __init__(self, in_channels: int, device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.device = device

        # 初始化模型并将模型移动到指定设备
        self.model = UNetBase(in_channels)
        self.model.to(self.device)

    def analyze(self, input_tensor_size: tuple) -> None:
        """
        根据输入张量尺寸计算模型的 FLOPs 和参数数量，并打印结果。

        参数:
        input_tensor_size (tuple): 输入张量尺寸，格式为 (Channels, Height, Width)
        """
        # 构造一个与模型输入匹配的随机张量 (Batch size 默认为 1)
        input_tensor = torch.randn(1, *input_tensor_size).to(self.device)

        # 使用 FlopCountAnalysis 计算 FLOPs
        flops = FlopCountAnalysis(self.model, input_tensor)
        print("Total FLOPs:", flops.total())

        # 输出模型参数数量统计
        print("\nParameters:")
        print(parameter_count_table(self.model))