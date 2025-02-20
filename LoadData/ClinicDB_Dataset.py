import numpy as np
import os

import tifffile as tiff
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from LoadData.assistance import build_transforms


class ClinicDB_Dataset(Dataset):
    """
    ClinicDB Dataset. 读取文件，返回文件的tensor格式
    """
    def __init__(self, config, augmentations, transform_label=None, class_num=1):
        self.config = config
        self.image_dir = os.path.join(self.config["dataset_path"], self.config["img"])
        self.image_names =[f for f in os.listdir(self.image_dir) if f.endswith('.tif')]

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
        image_name = self.image_names[index]
        mask_name = self.mask_names[index]

        mask_path = os.path.join(self.mask_dir, mask_name)
        image_path = os.path.join(self.image_dir, image_name)

        # 读取TIF图像
        image = tiff.imread(image_path)
        mask = tiff.imread(mask_path)

        # 处理掩码维度
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
        mask = mask.squeeze()  # 从 (H, W, 1) 转为 (H, W)

        # 转换为PIL.Image
        # 确保图像数据为uint8类型（假设原始数据范围0-255）
        image_pil = Image.fromarray(image.astype(np.uint8))
        # 处理掩码数据（假设二值掩码0/1，转换为0/255）
        mask = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask, mode='L')  # 'L'模式表示8位灰度

        # 应用数据增强
        image_aug, mask_aug = self.transforms(image=image_pil, mask=mask_pil)

        # 转换为Tensor
        image_tensor = transforms.ToTensor()(image_aug)
        mask_tensor = transforms.ToTensor()(mask_aug)  # 自动转换为[C, H, W]

        # 可选标签变换
        if self.transform_label:
            mask_tensor = self.transform_label(mask_tensor)

        return image_tensor, mask_tensor