import random
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image


class SynchronizedTransform:
    """
    自定义同步变换类，确保图像和 mask 同时进行相同的增强操作。
    """

    def __init__(self, transforms):
        self.transforms = transforms  # 传入需要同步的增强操作列表

    def __call__(self, image, mask=None):
        """
        对图像和 mask 同时应用增强。
        """
        for transform in self.transforms:
            seed = random.randint(0, 2 ** 32)  # 确保随机数种子一致
            random.seed(seed)
            torch.manual_seed(seed)

            # 确保变换对图像和 mask 同步应用
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
