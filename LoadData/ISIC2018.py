import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, to_pil_image

# 定义图像和标签的变换
transform_image = transforms.Compose([
    transforms.ToTensor()
])  # 图像转换为张量


# 保持图像大小的打开方法
def keep_image_size_open(file_path, size):
    image = Image.open(file_path).convert('RGB')  # 强制为 RGB 格式
    image = image.resize(size, Image.BILINEAR)
    return image


class ISIC2018_DataSet(Dataset):
    """
    自定义数据集，加载图像和对应的标签
    """
    def __init__(self, config, _transform_image=None, transform_label=None, class_num=1):
        self.config = config
        self.mask_name = os.listdir(os.path.join(self.config["dataset_path"], self.config["mask"]))
        self.transform_image = _transform_image
        self.transform_label = transform_label
        self.class_num = class_num

    def __len__(self):
        return len(self.mask_name)

    def __getitem__(self, index):
        # 获取掩膜文件名和路径
        segment_name = self.mask_name[index]
        segment_path = os.path.join(self.config["dataset_path"], self.config['mask'], segment_name)

        # 根据掩膜文件名生成图像文件名
        image_name = segment_name.replace(self.config["seg_prefix"], self.config["img_prefix"]).replace(
            self.config["seg_suffix"], self.config["img_suffix"])
        image_path = os.path.join(self.config["dataset_path"], self.config["img"], image_name)

        # 加载图像和标签
        segment_image = keep_image_size_open(segment_path, (self.config["size"], self.config["size"]))
        img_image = keep_image_size_open(image_path, (self.config["size"], self.config["size"]))

        # 应用图像和标签的变换
        if self.transform_image:
            img_image = self.transform_image(img_image)
        if self.transform_label:
            segment_image = self.transform_label(transform_image(segment_image))

        return img_image, segment_image


# 根据 augmentations 配置生成动态的 transforms
def get_transforms(augmentations):
    transforms_list = []
    for aug in augmentations:
        # 如果 aug 是字典，按配置中的参数解析
        if aug["type"] == "RandomHorizontalFlip":
            transforms_list.append(transforms.RandomHorizontalFlip(p=aug.get("p", 0.5)))
        elif aug["type"] == "RandomRotation":
            transforms_list.append(transforms.RandomRotation(degrees=aug.get("degrees", 15)))
        elif aug["type"] == "Resize":
            transforms_list.append(transforms.Resize(size=aug.get("size", (256, 256))))
        elif aug["type"] == "ColorJitter":
            transforms_list.append(transforms.ColorJitter(
                brightness=aug.get("brightness", None),  # 使用 None 作为默认值
                contrast=aug.get("contrast", None),
                saturation=aug.get("saturation", None),
                hue=aug.get("hue", None)
            ))
        else:
            raise ValueError(f"Unsupported augmentation type: {aug['type']}")
    return SynchronizedTransform(transforms_list)


class SynchronizedTransform:
    """
    自定义同步变换类，确保图像和 mask 同时进行相同的增强操作。
    """

    def __init__(self, _transforms):
        self.transforms = _transforms  # 传入需要同步的增强操作列表

    def __call__(self, image, mask=None):
        """
        对图像和 mask 同时应用增强。
        """
        for transform in self.transforms:
            seed = random.randint(0, 2 ** 32)  # 确保随机数种子一致
            random.seed(seed)
            torch.manual_seed(seed)

            # 检查变换类型
            if isinstance(transform, transforms.ColorJitter):
                # 色彩变换只应用于图像
                image = transform(image)
            else:
                # 几何变换同步应用
                image = transform(image)
                if mask is not None:
                    random.seed(seed)
                    torch.manual_seed(seed)
                    mask = transform(mask)

        return image, mask


class AugmentedDataset(torch.utils.data.Dataset):
    """
    包装数据集类，确保图像和 mask 同时进行增强。
    """

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 获取原始数据
        sample = self.base_dataset[idx]

        image, label = sample

        # 将图像和 mask 转换为 PIL 格式以应用变换
        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)

        # 应用同步增强
        if self.transform:
            image, label = self.transform(image, label)

        # 转回 Tensor 格式
        image = to_tensor(image)

        return (image, label)
