# 定义一个模型加载函数
import os
import torch
from glob import glob
from LoadData.utils import load_config
from model_defination.AAA_BNet.BNet import BNet
from model_defination.ResNet.resnet import ResNet101, ResNet50, ResNet152
from model_defination.AAA_Unet.unet import UNetBase
from model_defination.fcn_8s.fnc_8s import FCN8s
from model_defination.AAA_unetpp.unetpp import UnetPP

# load the config file
CONFIG_NAME = "config_train.yaml"
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


# 模型映射表
model_mapping = {
    # TODO：模型我想添加
    # TODO：duck-net https://github.com/RazvanDu/DUCK-Net
    # TODO: nn-unet https://github.com/MIC-DKFZ/nnUNet
    "bnet": lambda: BNet(1),
    "unet": lambda: UNetBase(1),
    "unetpp": lambda: UnetPP(1),

    "res50": lambda: ResNet50(1),
    "res101": lambda: ResNet101(1),
    "res152": lambda: ResNet152(1),
    "fcn_8s": lambda: FCN8s(1),

}


# 定义一个模型加载函数
def load_model(_config, mode):
    """
    根据配置文件中的模型名称加载模型结构，并返回模型实例。

    :param _config: 配置字典，包含模型名称、路径等信息。
    :param mode: 加载模式，'train' 或 'test'，用于区分加载训练或测试权重。
    :return: 初始化的模型实例。
    """
    model_name = _config.get("model")['name']
    model_path = _config.get("model")['save_path']

    # 初始化模型
    if model_name not in model_mapping:
        raise ValueError(f"Unknown model name '{model_name}' in config file.")
    model = model_mapping[model_name]()

    # 如果是训练模式，尝试加载权重
    if mode == "train":
        try:
            weight_path = get_best_or_latest_model_path(model_path, model_name)
            print(f"Loading weights from {weight_path}")
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            print("Successfully loaded weights.")
        except FileNotFoundError as e:
            print(e)
            print("No weights loaded.")

    return model.to(device)
