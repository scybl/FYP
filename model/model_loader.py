import torch
from model.FYPNet.FYPNet_Res34 import BNet_Res34
from model.FYPNet.FYPNet import FYPNet
from model.DuckNet.DuckNet import DuckNet
from model.Unet.unet import UNetBase
from model.UNeXt.unext import UNext
from model.UnetPP.unetpp import UnetPP
from model.FYPNet.DGANet import DGANet
from model.FYPNet.PHAMNet import PHAMNet

import os
from glob import glob
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def get_best_or_latest_model_path(model_path, model_name, dataset_name):
    f"""
        Load model path according to the following rules:
        1. Prioritize loading "{model_name}_{dataset_name}_best.pth"
        2. If "_best" is not found, load the one with the highest number "{model_name}_{dataset_name}_NO.pth"
    """
    model_files = glob(os.path.join(model_path, f"{model_name}_{dataset_name}_*.pth"))

    if not model_files:
        raise FileNotFoundError(f"No model files found for {model_name} in {model_path}")

    best_file = next((f for f in model_files if f"{model_name}_{dataset_name}_best.pth" in f), None)
    if best_file:
        return best_file

    numbered_files = [
        (f, int(f.split("_")[-1].split(".")[0])) for f in model_files if f"{model_name}_{dataset_name}_best" not in f
    ]
    if not numbered_files:
        raise FileNotFoundError(f"No valid numbered model files found for {model_name} in {model_path}")

    numbered_files.sort(key=lambda x: x[1], reverse=True)
    return numbered_files[0][0]


def get_model_hub(in_channel, class_num):
    model_hub = {
        "bnet34": lambda: BNet_Res34(in_channel=in_channel, num_classes=class_num, encoder_mode='res34',
                                     pre_train=True),
        "bnet": lambda: FYPNet(in_channel=in_channel, num_classes=class_num),
        "unet": lambda: UNetBase(in_channel=in_channel, class_num=class_num),

        "unetpp": lambda: UnetPP(in_channel, num_classes=class_num, deep_supervision=False),
        "duck": lambda: DuckNet(in_channel=in_channel, num_classes=class_num),

        'unext': lambda: UNext(num_classes=class_num, input_channels=in_channel, deep_supervision=True),
        'pham': lambda: PHAMNet(num_classes=class_num, in_channel=in_channel, deep_supervisor=False),
        'dga': lambda: DGANet(num_classes=class_num, in_channel=in_channel, deep_supervisor=False),
    }
    return model_hub


def load_model(_config, _model_name, dataset_name):
    """
        Load the model structure based on the model name in the configuration file and return the model instance.

        :param _config: Configuration dictionary containing model name, path, and other information.
        :return: Initialized model instance.
    """
    model_path = _config.get("model")['save_path']

    dataset_name = dataset_name

    in_channel = _config["datasets"][dataset_name]["in_channel"]
    class_num = _config["datasets"][dataset_name]['class_num']

    model_hub = get_model_hub(in_channel=in_channel, class_num=class_num)
    device = _config["device"]
    if _model_name not in model_hub:
        raise ValueError(f"Unknown model name '{_model_name}' in config file.")

    model = model_hub[_model_name]()

    try:
        weight_path = get_best_or_latest_model_path(model_path, _model_name, dataset_name)
        print(f"Loading weights from {weight_path}")

        check_point = torch.load(weight_path, map_location=device, weights_only=True)
        load_result = model.load_state_dict(check_point, strict=False)

        if load_result.missing_keys or load_result.unexpected_keys:
            logging.error(
                f"Model loading issues detected. Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")
        else:
            print("Successfully loaded weights.")

    except Exception as e:
        logging.error(f"Error loading weights: {e}")

    return model.to(device)
