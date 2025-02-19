import os
from torch.utils.data import Dataset
import random
import torch
from PIL import Image
import torchvision.transforms as transforms

# 定义图像和标签的变换
transform_image = transforms.Compose([
    transforms.ToTensor()
])  # 图像转换为张量


# 统一加载图像方法
def load_image(file_path, size=None):
    image = Image.open(file_path).convert('RGB')
    if size:
        image = image.resize(size, Image.BILINEAR)
    return image


class SynchronizedTransform:
    """确保图像和 mask 同时进行相同增强操作"""

    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, mask=None):
        seed = random.randint(0, 2 ** 32)  # 生成一次随机种子
        random.seed(seed)
        torch.manual_seed(seed)

        for transform in self.transforms:
            if isinstance(transform, transforms.ColorJitter):
                image = transform(image)  # 色彩变换只应用于图像
            else:
                image = transform(image)
                if mask is not None:
                    random.seed(seed)  # 确保 mask 变换一致
                    torch.manual_seed(seed)
                    mask = transform(mask)

        return image, mask


def build_transforms(augmentations):
    """根据 augmentations 生成增强变换"""
    transform_list = []

    if augmentations.get("geometric_transforms") is not None:
        # 解析几何变换
        for aug in augmentations.get("geometric_transforms"):
            transform_type = aug["type"]
            params = aug.get("params", {})

            if transform_type == "RandomHorizontalFlip":
                transform_list.append(transforms.RandomHorizontalFlip(p=params.get("p", 0.5)))
            elif transform_type == "RandomRotation":
                transform_list.append(transforms.RandomRotation(degrees=params.get("degree", 15)))
            elif transform_type == "Resize":
                transform_list.append(transforms.Resize(size=params.get("size", (256, 256))))
            else:
                raise ValueError(f"Unsupported geometric transformation: {transform_type}")

        # 解析色彩变换
        color_params = {aug["type"]: aug["params"]["degree"] for aug in augmentations.get("color_transforms", [])}
        if color_params:
            transform_list.append(transforms.ColorJitter(
                brightness=color_params.get("brightness"),
                contrast=color_params.get("contrast"),
                saturation=color_params.get("saturation"),
                hue=color_params.get("hue")
            ))

    return SynchronizedTransform(transform_list)








class ISIC2018_DataSet(Dataset):
    """
    自定义数据集，加载图像和对应的标签，同时应用同步数据增强
    """

    def __init__(self, config, augmentations, transform_label=None, class_num=1):
        self.config = config
        self.mask_name = os.listdir(os.path.join(self.config["dataset_path"], self.config["mask"]))
        self.transform_label = transform_label  # 保留 transform_label
        self.class_num = class_num

        # **使用 SynchronizedTransform 进行同步数据增强**
        self.transforms = build_transforms(augmentations)

        # **确保最终数据转换为 Tensor**
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.mask_name)

    def __getitem__(self, index):
        # 获取 mask 文件名及路径
        segment_name = self.mask_name[index]
        segment_path = os.path.join(self.config["dataset_path"], self.config["mask"], segment_name)

        # 生成对应的 image 文件名及路径
        image_name = segment_name.replace(self.config["seg_prefix"], self.config["img_prefix"]).replace(
            self.config["seg_suffix"], self.config["img_suffix"])
        image_path = os.path.join(self.config["dataset_path"], self.config["img"], image_name)

        # **加载图像 (RGB)**
        img_image = Image.open(image_path).convert("RGB")  # 确保 image 为 3 通道
        segment_image = Image.open(segment_path).convert("L")  # **转换为灰度模式，确保单通道**

        # **同步几何变换**
        img_image, segment_image = self.transforms(img_image, segment_image)

        # **对 mask 进行 transform_label 额外处理**
        if self.transform_label:
            segment_image = self.transform_label(segment_image)

        # **转换为 Tensor**
        img_image = self.to_tensor(img_image)  # 变为 (3, H, W)
        segment_image = self.to_tensor(segment_image)  # **变为 (1, H, W)，避免通道不匹配**
        return img_image, segment_image
