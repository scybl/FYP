from model_defination.AAA_Unet.unet import UNetBase
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table

"""
参数量主要影响-训练时间，因为它决定了模型在反向传播中需要优化的权重数量。
计算量主要影响-推理时间，因为它决定了模型在正向传播中需要执行的计算操作数量。
"""

# 定义模型
model = UNetBase(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 输入张量大小 (Batch size, Channels, Height, Width)
input_size = (3, 256, 256)

# 创建一个与模型输入相匹配的随机 tensor
input_tensor = torch.randn(1, 3, 224, 224)

# 使用 FlopCountAnalysis 计算 FLOPs
flops = FlopCountAnalysis(model, input_tensor)
print("Total FLOPs:", flops.total())

# 输出详细的 FLOPs 统计表
# print("\nFLOPs breakdown:")
# print(flop_count_table(model, input_tensor))

# 使用 parameter_count_table 输出模型参数数量统计
print("\nParameters:")
print(parameter_count_table(model))
