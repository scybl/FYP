from ptflops import get_model_complexity_info
import torch
from model_defination.MyFrame.UnetFrame import BNet

"""
参数量主要影响-训练时间，因为它决定了模型在反向传播中需要优化的权重数量。
计算量主要影响-推理时间，因为它决定了模型在正向传播中需要执行的计算操作数量。
"""
# 定义模型
model = BNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 输入张量大小 (Batch size, Channels, Height, Width)
input_size = (3, 256, 256)

# 打开文件写入模式
with open("output.txt", "w") as f:
    # 计算 FLOPs 和参数量
    flops, params = get_model_complexity_info(
        model, input_size, as_strings=True, print_per_layer_stat=True, ost=f
    )
    # 额外保存 FLOPs 和 Params 总量
    f.write(f"\nFLOPs: {flops}\n")
    f.write(f"Params: {params}\n")
