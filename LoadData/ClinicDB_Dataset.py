import os

import tifffile as tiff
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

from LoadData.utils import build_transforms


class ClinicDB_Dataset(Dataset):
    """
    ClinicDB Dataset. 读取文件，返回文件的tensor格式
    """

    def __init__(self, config):
        self.config = config
        self.image_dir = os.path.join(self.config["dataset_path"], self.config["img"])
        self.image_names = [f for f in os.listdir(self.image_dir) if f.endswith('.tif')]

        self.mask_dir = os.path.join(self.config["dataset_path"], self.config["mask"])
        self.mask_names = [f for f in os.listdir(self.mask_dir) if f.endswith('.tif')]  # 显式过滤TIF文件
        self.class_num = config["class_num"]

        self.transforms = build_transforms(config['augmentations'])

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.mask_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        mask_name = self.mask_names[index]

        mask_path = os.path.join(self.mask_dir, mask_name)
        image_path = os.path.join(self.image_dir, image_name)

        # 读取TIF图像
        image = tiff.imread(image_path) # ndarray [288,384,3] ,是一个三通道图片
        mask = tiff.imread(mask_path) # ndarray [288,384],是一个单通道图片

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        image,mask = self.transforms(image,mask)
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        return image, mask