import torch

from model_defination.UnetBase.unetbase import UNetBase
from model_defination.AAA_BNet.BNet import BNet


import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# 确保正确导入你的模型类
# from your_model_file import BNet, UNetBase

model_name = "UNet"

if model_name == "BNet":
    model = BNet()
    model.load_state_dict(torch.load("params/bnet_1.pth", map_location=torch.device('cpu')))

elif model_name == "UNet":
    model = UNetBase()
    model.load_state_dict(torch.load("params/unet0_1.pth", map_location=torch.device('cpu')))

# 计算参数数量
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters in {model_name}: {num_params} ({num_params / 1e6:.2f}M)")

# 计算 FLOPs
input_tensor = torch.randn(1, 3, 256, 256)  # 适应你的输入尺寸
flops = FlopCountAnalysis(model, input_tensor)

print(f"Total FLOPs in {model_name}: {flops.total() / 1e9:.2f} GFLOPs")