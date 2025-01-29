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
    """
    自定义同步变换类，确保图像和 mask 同时进行相同的增强操作。
    """

    def __init__(self, _transforms):
        self.transforms = _transforms

    def __call__(self, image, mask=None):
        for transform in self.transforms:
            seed = random.randint(0, 2 ** 32)  # 保持变换同步
            random.seed(seed)
            torch.manual_seed(seed)

            if isinstance(transform, transforms.ColorJitter):
                image = transform(image)  # 色彩变换只应用于图像
            else:
                image = transform(image)
                if mask is not None:
                    random.seed(seed)
                    torch.manual_seed(seed)
                    mask = transform(mask)

        return image, mask


# 根据 augmentations 配置生成动态的 transforms
def geometric(augmentations):
    transforms_list = []
    for aug in augmentations:
        # 如果 aug 是字典，按配置中的参数解析
        if aug["type"] == "RandomHorizontalFlip":
            transforms_list.append(transforms.RandomHorizontalFlip(p=aug.get("p", 0.5)))
        elif aug["type"] == "RandomRotation":
            transforms_list.append(transforms.RandomRotation(degrees=aug.get("degrees", 15)))
        elif aug["type"] == "Resize":
            transforms_list.append(transforms.Resize(size=aug.get("size", (256, 256))))
        else:
            raise ValueError(f"Unsupported augmentation type: {aug['type']}")
    return SynchronizedTransform(transforms_list)

def color_trans(augumentations):

