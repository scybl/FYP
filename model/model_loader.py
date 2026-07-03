import torch
import os
from glob import glob
import logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


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

    numbered_files = []
    for f in model_files:
        file_name = os.path.splitext(os.path.basename(f))[0]
        if file_name == f"{model_name}_{dataset_name}_best":
            continue
        number = file_name.rsplit("_", 1)[-1]
        if number.isdigit():
            numbered_files.append((f, int(number)))

    if not numbered_files:
        raise FileNotFoundError(f"No valid numbered model files found for {model_name} in {model_path}")

    numbered_files.sort(key=lambda x: x[1], reverse=True)
    return numbered_files[0][0]


def get_model_hub(in_channel, class_num, pretrained_encoder=False):
    def make_bnet34():
        from model.FYPNet.FYPNet_Res34 import BNet_Res34
        return BNet_Res34(in_channel=in_channel, num_classes=class_num, encoder_mode='res34',
                          pre_train=pretrained_encoder)

    def make_bnet():
        from model.FYPNet.FYPNet import FYPNet
        return FYPNet(in_channel=in_channel, num_classes=class_num)

    def make_unet():
        from model.Unet.unet import UNetBase
        return UNetBase(in_channel=in_channel, class_num=class_num)

    def make_unetpp():
        from model.UnetPP.unetpp import UnetPP
        return UnetPP(in_channel, num_classes=class_num, deep_supervision=False)

    def make_duck():
        from model.DuckNet.DuckNet import DuckNet
        return DuckNet(in_channel=in_channel, num_classes=class_num)

    def make_unext():
        try:
            from model.UNeXt.unext import UNext
        except ModuleNotFoundError as exc:
            if exc.name == "timm":
                raise ModuleNotFoundError("UNeXt requires timm. Please install requirements.txt before using unext.") from exc
            raise
        return UNext(num_classes=class_num, input_channels=in_channel, deep_supervision=True)

    def make_pham():
        from model.FYPNet.PHAMNet import PHAMNet
        return PHAMNet(num_classes=class_num, in_channel=in_channel, deep_supervisor=False)

    def make_dga():
        from model.FYPNet.DGANet import DGANet
        return DGANet(num_classes=class_num, in_channel=in_channel, deep_supervisor=False)

    model_hub = {
        "bnet34": make_bnet34,
        "bnet": make_bnet,
        "unet": make_unet,
        "unetpp": make_unetpp,
        "duck": make_duck,
        "unext": make_unext,
        "pham": make_pham,
        "dga": make_dga,
    }
    return model_hub


def resolve_device(device_name):
    if isinstance(device_name, str) and device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu"
        if ":" in device_name:
            device_index = int(device_name.split(":", 1)[1])
            if device_index >= torch.cuda.device_count():
                return "cuda:0"
    return device_name


def load_model(_config, _model_name, dataset_name, load_weights=True):
    """
        Load the model structure based on the model name in the configuration file and return the model instance.

        :param _config: Configuration dictionary containing model name, path, and other information.
        :return: Initialized model instance.
    """
    model_path = _config.get("model")['save_path']

    dataset_name = dataset_name

    in_channel = _config["datasets"][dataset_name]["in_channel"]
    class_num = _config["datasets"][dataset_name]['class_num']

    pretrained_encoder = _config.get("model", {}).get("pretrained_encoder", False)
    model_hub = get_model_hub(
        in_channel=in_channel,
        class_num=class_num,
        pretrained_encoder=pretrained_encoder,
    )
    device = resolve_device(_config.get("device", "cpu"))
    if _model_name not in model_hub:
        raise ValueError(f"Unknown model name '{_model_name}' in config file.")

    model = model_hub[_model_name]()

    if not load_weights:
        return model.to(device)

    try:
        weight_path = get_best_or_latest_model_path(model_path, _model_name, dataset_name)
        print(f"Loading weights from {weight_path}")

        try:
            check_point = torch.load(weight_path, map_location=device, weights_only=True)
        except TypeError:
            check_point = torch.load(weight_path, map_location=device)
        load_result = model.load_state_dict(check_point, strict=False)

        if load_result.missing_keys or load_result.unexpected_keys:
            logging.error(
                f"Model loading issues detected. Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")
        else:
            print("Successfully loaded weights.")

    except FileNotFoundError as e:
        logging.warning(f"{e}. Use initialized weights instead.")
    except Exception as e:
        logging.error(f"Error loading weights: {e}")

    return model.to(device)
