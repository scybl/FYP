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

    def __init__(self, config, mode):
        self.config = config
        self.class_num = config["class_num"]
        self.dataset_path = config["dataset_path"]

        if mode == "train":
            self.img_path = os.path.join(self.dataset_path, self.config["train_img"])
            self.mask_path = os.path.join(self.dataset_path, self.config["train_mask"])
        elif mode =="val":
            self.img_path = os.path.join(self.dataset_path, self.config["val_img"])
            self.mask_path = os.path.join(self.dataset_path, self.config["val_mask"])
        elif mode == 'test':
            self.img_path =os.path.join(self.dataset_path, self.config["test_img"])
            self.mask_path = os.path.join(self.dataset_path, self.config["test_mask"])

        self.img_list = os.listdir(self.img_path)
        self.mask_list = os.listdir(self.mask_path)

        # **使用 SynchronizedTransform 进行同步数据增强**
        self.transforms = build_transforms(config['augmentations'])

        # **确保最终数据转换为 Tensor**
        self.to_tensor = transforms.ToTensor()
        self.transform_label = None


    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_file_name = self.mask_list[index]
        img_file_name = self.img_list[index]

        img_path = os.path.join(self.img_path, img_file_name)
        mask_path = os.path.join(self.mask_path, mask_file_name)

        # 读取TIF图像
        image = tiff.imread(img_path) # ndarray [288,384,3] ,是一个三通道图片
        mask = tiff.imread(mask_path) # ndarray [288,384],是一个单通道图片

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        image,mask = self.transforms(image,mask)
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        return image, mask