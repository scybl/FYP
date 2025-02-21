import numpy as np
import os

import tifffile as tiff
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from LoadData.assistance import build_transforms
import torch

"""
添加synapse数据集
"""
# 颜色映射表
color_mapping = {
    (0, 0, 0): 0, # 黑色
    (255, 255, 255): 1, # 白色
    (0, 0, 255): 2, # 红色
    (0, 255, 0): 3, # 绿色
    (255, 0, 0): 4, # 蓝色
    (0, 255, 255): 5, # 黄色
    (255, 0, 255): 6, # 紫色
    (255, 255, 0): 7 # 青色
}


class Synapse_Dataset(Dataset):

    def __init__(self, config, augmentations, class_num=8):
        self.config = config
        self.image_dir = os.path.join(self.config["dataset_path"], self.config["img"])
        self.image_names = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]

        self.mask_dir = os.path.join(self.config["dataset_path"], self.config["mask"])
        self.mask_names = [f for f in os.listdir(self.mask_dir) if f.endswith('.png')]  # 显式过滤TIF文件

        self.class_num = class_num

        # 同步数据增强组件
        self.transforms = build_transforms(augmentations)

        # TIF特殊处理：可能需要添加Alpha通道处理（根据实际数据情况）
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.mask_names)

    def __getitem__(self, index):

        # 获取图像和分割标签路径
        segment_name = self.mask_names[index]
        segment_path = os.path.join(self.mask_dir, segment_name)
        image_path = os.path.join(self.image_dir, segment_name)

        # **加载原始图像**
        img_image = Image.open(image_path).convert("RGB")  # 确保 image 为 3 通道
        segment_image = Image.open(segment_path).convert("RGB")  # 以 RGB 加载分割图像\

        img_image, segment_image = self.transforms(img_image, segment_image) # 数据增强
        # 加载的图片是没有问题的

        # **转换为 Tensor**
        img_image = self.to_tensor(img_image)  # 变为 (3, H, W)

        # TODO: 在这里根据color map将segment_image从<PIL.Image.Image image mode=RGB size=512x512>
        #  TODO: 映射为tensor (8, H, W)，

        # 转换segment_image为one-hot tensor
        segment_array = np.array(segment_image)  # (H, W, 3)

        # 创建初始mask
        h, w = segment_array.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # 根据颜色映射填充类别
        for color, class_id in color_mapping.items():
            # 使用向量化操作提升性能
            mask[np.all(segment_array == color, axis=-1)] = class_id

        # 转换为tensor并生成one-hot编码
        mask_tensor = torch.from_numpy(mask).long()
        one_hot = torch.nn.functional.one_hot(
            mask_tensor,
            num_classes=self.class_num
        ).permute(2, 0, 1).float()  # (C, H, W)

        print(one_hot)

        return img_image, one_hot  # 形状: (3, H, W), (8, H, W)