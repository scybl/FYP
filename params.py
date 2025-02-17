from model_defination.AAA_BNet.BNet import BNet
import torch

model = BNet()
model.load_state_dict(torch.load("bnet.pth", map_location=torch.device('cpu')))

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters in BNet: {num_params}")
