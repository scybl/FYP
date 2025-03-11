import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms

from LoadData.utils import build_transforms


class KvasirSEG_Dataset(data.Dataset):

    def __init__(self, config, mode):
        self.config = config
        self.class_num = config["class_num"]
        self.dataset_path = self.config["dataset_path"]

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
        # 获取 mask 文件名及路径
        mask_file_name = self.mask_list[index]

        mask_file = os.path.join(self.mask_path, mask_file_name)

        # 生成对应的 image 文件名及路径
        image_name = mask_file_name

        image_path = os.path.join(self.img_path, image_name)

        # **加载图像 (RGB)**
        img_image = Image.open(image_path).convert("RGB")  # 确保 image 为 3 通道
        mask_image = Image.open(mask_file).convert("L")  # **转换为灰度模式，确保单通道**

        # **同步几何变换**
        img_image, mask_image = self.transforms(img_image, mask_image)

        # **对 mask 进行 transform_label 额外处理**
        if self.transform_label:
            mask_image = self.transform_label(mask_image)

        # **转换为 Tensor**
        img_image = self.to_tensor(img_image)  # 变为 (3, H, W)
        mask_image = self.to_tensor(mask_image)  # **变为 (1, H, W)，避免通道不匹配**
        return img_image, mask_image

