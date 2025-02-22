import random
import torch
import torchvision.transforms as transforms


class SynchronizedTransform:
    """确保图像和 mask 同时进行相同的增强操作"""

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
