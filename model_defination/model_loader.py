# 定义一个模型加载函数
import os
import torch
from glob import glob
from LoadData.utils import load_config
from model_defination.MyFrame.UnetFrame import BNet
from model_defination.ResNet.resnet import ResNet101, ResNet50, ResNet152
from model_defination.UnetBase.unetbase import UNetBase
from model_defination.fcn_8s.fnc_8s import FCN8s
from model_defination.unetpp.unetpp import UnetPP

# load the config file
CONFIG_NAME = "config.yaml"
CONFIG_PATH = os.path.join("configs/", CONFIG_NAME)
config = load_config(CONFIG_PATH)

device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")


def get_best_or_latest_model_path(model_path, model_name):
    """
    根据规则加载模型路径:
    1. 优先加载 "{model_name}_best.pth"
    2. 如果没有 "_best"，加载编号最大的 "{model_name}_NO.pth"
    """
    # 获取路径中所有符合命名规则的文件
    model_files = glob(os.path.join(model_path, f"{model_name}_*.pth"))

    if not model_files:
        raise FileNotFoundError(f"No model files found for {model_name} in {model_path}")

    # 优先选择 `_best` 文件
    best_file = next((f for f in model_files if f"{model_name}_best.pth" in f), None)
    if best_file:
        return best_file

    # 如果没有 `_best`，按编号从大到小排序选择最大编号
    numbered_files = [
        (f, int(f.split("_")[-1].split(".")[0])) for f in model_files if f"{model_name}_best" not in f
    ]
    if not numbered_files:
        raise FileNotFoundError(f"No valid numbered model files found for {model_name} in {model_path}")

    # 根据编号排序，选择最大的编号
    numbered_files.sort(key=lambda x: x[1], reverse=True)
    return numbered_files[0][0]


# 定义一个模型加载函数
def load_model_train(_config):
    model_name = _config.get("model_name")
    model_path = _config.get("model_path")

    # 根据模型名初始化模型
    if model_name == "unet0":
        _model = UNetBase(1)
    elif model_name == "unetPP":
        _model = UnetPP(1)
    elif model_name == "res50":
        _model = ResNet152(1)
    elif model_name == "res101":
        _model = ResNet152(1)
    elif model_name == "res152":
        _model = ResNet152(1)
    elif model_name == "fcn_8s":
        _model = FCN8s(1)
    elif model_name == "bnet":
        _model = BNet(1)
    else:
        raise ValueError(f"Unknown model name '{model_name}' in config file.")

    # 获取权重文件路径
    try:
        weight_path = get_best_or_latest_model_path(model_path, model_name)
        print(f"Loading weights from {weight_path}")
        _model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        print("Successfully loaded weights.")
    except FileNotFoundError as e:
        print(e)
        print("No weights loaded.")

    return _model.to(device)


def load_model_test(_config):
    """
    根据配置文件中的模型名称加载模型结构，并返回模型实例。
    """
    model_name = _config.get("model_name")

    if model_name == "unet0":
        model = UNetBase()
    elif model_name == "unetPP":
        model = UnetPP()
    elif model_name == "res50":
        model = ResNet50()
    elif model_name == "res101":
        model = ResNet101()
    elif model_name == "res152":
        model = ResNet152()
    elif model_name == "fcn_8s":
        model = FCN8s()
    else:
        raise ValueError(f"Unknown model name '{model_name}' in config file.")

    return model


model_total = {
    "unetO": UNetBase,  # this is the basic unet model
    "unetPP": UnetPP,
    "res50": ResNet50,
    "res101": ResNet101,
    "res152": ResNet152,
    "fcn_8s": FCN8s,
    "bnet": BNet,

}
