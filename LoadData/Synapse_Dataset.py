import numpy as np
import os

import tifffile as tiff
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from LoadData.assistance import build_transforms


"""
添加synapse数据集
"""
from LoadData.assistance import build_transforms


class ClinicDB_Dataset(Dataset):

    def __init__(self, config, augmentations, transform_label=None, class_num=1):
        self.config = config
        self.image_dir = os.path.join(self.config["dataset_path"], self.config["img"])
        self.image_names = [f for f in os.listdir(self.image_dir) if f.endswith('.tif')]

        self.mask_dir = os.path.join(self.config["dataset_path"], self.config["mask"])
        self.mask_names = [f for f in os.listdir(self.mask_dir) if f.endswith('.tif')]  # 显式过滤TIF文件
        self.transform_label = transform_label
        self.class_num = class_num

        # 同步数据增强组件
        self.transforms = build_transforms(augmentations)

        # TIF特殊处理：可能需要添加Alpha通道处理（根据实际数据情况）
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.mask_names)

    def __getitem__(self, index):
