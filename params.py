import torch

from model_defination.UnetBase.unetbase import UNetBase
from model_defination.AAA_BNet.BNet import BNet


model_name = "UNet"

if model_name == "BNet":
    model = BNet()
    model.load_state_dict(torch.load("params/bnet_1.pth", map_location=torch.device('cpu')))


elif model_name == "UNet": 
    model = UNetBase()
    model.load_state_dict(torch.load("params/unet0_1.pth", map_location=torch.device('cpu')))



num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters in BNet: {num_params}")