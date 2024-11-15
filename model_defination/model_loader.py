import os
import torch

from LoadData.utils import load_config
from model_defination.ResNet.resnet import ResNet101, ResNet50, ResNet152
from model_defination.UnetBase.unetbase import UNetBase
from model_defination.fcn_8s.fnc_8s import FCN8s
from model_defination.unetpp.unetpp import UnetPP

# load the config file
CONFIG_NAME = "config.yaml"
CONFIG_PATH = os.path.join("../configs/", CONFIG_NAME)
config = load_config(CONFIG_PATH)

device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
weight_path = os.path.join(config['model_path'], config['model_name'] + ".pth")


# 定义一个模型加载函数
def load_model(config):
    model_name = config.get("model_name")

    if model_name == "unet":
        model = UNetBase()
    elif model_name == "unetpp":
        model = UnetPP()
    elif model_name == "res50":
        model = ResNet152()
    elif model_name == "res101":
        model = ResNet152()
    elif model_name == "res152":
        model = ResNet152()
    elif model_name == "fcn_8s":
        model = FCN8s()

    else:
        raise ValueError(f"Unknown model name '{model_name}' in config file.")

    if os.path.exists(os.path.join(os.getcwd(), weight_path)):
        print(f"Loading weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        print("successfully load weights")
    else:
        print("no successfully open weight")

    return model.to(device)


model_total = {
    "unet": UNetBase,  # this is the basic unet model
    "res50": ResNet50,  # ResNet50, ResNet101
    "res101": ResNet101,
    "res152": ResNet152,  # ResNet50, ResNet101
    "fcn_8s": FCN8s,
    "unetpp": UnetPP
}
