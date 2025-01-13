import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class DRIVEDataset(Dataset):
    """
    DRIVE 数据集加载器。
    """
    def __init__(self, root_dir, mode="train", transform=None):
        """
        :param root_dir: 数据集根目录，例如 "path/to/DRIVE"
        :param mode: "train" 或 "test"，决定加载训练或测试数据
        :param transform: 数据增强和预处理操作
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        # 确定当前模式的目录
        self.image_dir = os.path.join(root_dir, mode, "images")
        self.label_dir = os.path.join(root_dir, mode, "1st_manual")
        self.mask_dir = os.path.join(root_dir, mode, "mask")

        # 加载所有图像的文件名
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.label_filenames = sorted(os.listdir(self.label_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # 加载图像
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # 打开图像
        image = Image.open(image_path).convert("RGB")  # 转为 RGB 格式
        label = Image.open(label_path).convert("L")    # 转为灰度图
        mask = Image.open(mask_path).convert("L")      # 转为灰度图

        # 应用预处理或增强
        if self.transform:
            image, label, mask = self.transform(image, label, mask)

        # 转换为张量
        image = transforms.ToTensor()(image)
        label = torch.tensor(np.array(label) // 255, dtype=torch.float32)  # 二值化
        mask = torch.tensor(np.array(mask) // 255, dtype=torch.float32)   # 二值化

        return image, label, mask


