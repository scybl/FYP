# 定义一个模型加载函数
import torch
from model_defination.AAA_BNet.BNet_Res34 import BNet_Res34
from model_defination.AAA_BNet.Bnet import BNet
from model_defination.AAA_DuckNet.DuckNet import  DuckNet
from model_defination.AAA_Unet.unet import UNetBase
from model_defination.AAA_unetpp.unetpp import UnetPP

import os
from glob import glob
import logging

def get_best_or_latest_model_path(model_path, model_name, dataset_name):
    f"""
    根据规则加载模型路径:
    1. 优先加载 "{model_name}_{dataset_name}_best.pth"
    2. 如果没有 "_best"，加载编号最大的 "{model_name}_{dataset_name}_NO.pth"
    """
    # 获取路径中所有符合命名规则的文件
    model_files = glob(os.path.join(model_path, f"{model_name}_{dataset_name}_*.pth"))

    if not model_files:
        raise FileNotFoundError(f"No model files found for {model_name} in {model_path}")

    # 优先选择 "_best" 文件
    best_file = next((f for f in model_files if f"{model_name}_{dataset_name}_best.pth" in f), None)
    if best_file:
        return best_file

    # 如果没有 "_best"，按编号从大到小排序选择最大编号
    numbered_files = [
        (f, int(f.split("_")[-1].split(".")[0])) for f in model_files if f"{model_name}_{dataset_name}_best" not in f
    ]
    if not numbered_files:
        raise FileNotFoundError(f"No valid numbered model files found for {model_name} in {model_path}")

    # 根据编号排序，选择最大的编号
    numbered_files.sort(key=lambda x: x[1], reverse=True)
    return numbered_files[0][0]



def get_model_hub(in_channel,class_num):
    # 模型映射表
    model_hub = {
        # TODO：模型我想添加
        # TODO：duck-net https://github.com/RazvanDu/DUCK-Net
        # TODO: nn-unet https://github.com/MIC-DKFZ/nnUNet
        "bnet34": lambda: BNet_Res34(in_channel=in_channel, num_classes=class_num, encoder_mode='res34', pre_train=True),
        "bnet": lambda : BNet(in_channel=in_channel, num_classes=class_num),
        "unet": lambda: UNetBase(in_channel=in_channel,class_num=class_num),

        "unetpp": lambda: UnetPP(in_channel,num_classes=class_num, deep_supervision=False),
        "duck": lambda: DuckNet(in_channel=in_channel, num_classes=class_num),
    }
    return model_hub


def load_model(_config, _model_name, dataset_name):
    """
    根据配置文件中的模型名称加载模型结构，并返回模型实例。

    :param _config: 配置字典，包含模型名称、路径等信息。
    :return: 初始化的模型实例。
    """
    _model_name = _model_name
    model_path = _config.get("model")['save_path']

    dataset_name = dataset_name

    in_channel = _config["datasets"][dataset_name]["in_channel"]
    class_num = _config["datasets"][dataset_name]['class_num']

    model_hub = get_model_hub(in_channel=in_channel,class_num=class_num)
    device = _config["device"]
    # 初始化模型
    if _model_name not in model_hub:
        raise ValueError(f"Unknown model name '{_model_name}' in config file.")

    model = model_hub[_model_name]()


    try:
        weight_path = get_best_or_latest_model_path(model_path, _model_name, dataset_name)
        print(f"Loading weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        print("Successfully loaded weights.")
    except FileNotFoundError as e:
        print(e)
        print("No weights loaded.")

    return model.to(device)
