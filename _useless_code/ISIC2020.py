import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, to_pil_image

class ISIC2020_DataSet(Dataset):
    """
    自定义数据集，加载图像和对应的标签
    """
    def __init__(self, config, _transform_image=None, transform_label=None, class_num=1):
        self.config = config
        self.mask_name = os.listdir(os.path.join(self.config["dataset_path"], self.config["mask"]))
        self.transform_image = _transform_image
        self.transform_label = transform_label
        self.class_num = class_num