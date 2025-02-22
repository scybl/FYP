import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms
from LoadData.assistance import build_transforms


class KvasirSEG_Dataset(data.Dataset):

    def __init__(self, config, augmentations, class_num=1):
        self.config = config
        self.mask_name = os.listdir(os.path.join(self.config["dataset_path"], self.config["mask"]))
        self.class_num = class_num

        # **使用 SynchronizedTransform 进行同步数据增强**
        self.transforms = build_transforms(augmentations)

        # **确保最终数据转换为 Tensor**
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        # 返回数据集的大小
        return len(self.mask_name)

    def __getitem__(self, index):
        segment_name = self.mask_name[index]
        segment_path = os.path.join(self.config["dataset_path"], self.config["mask"], segment_name)

        # 生成对应的 image 文件名及路径
        image_name = segment_name
        image_path = os.path.join(self.config["dataset_path"], self.config["img"], image_name)

        # **加载图像 (RGB)**
        img_image = Image.open(image_path).convert("RGB")  # 确保 image 为 3 通道
        segment_image = Image.open(segment_path).convert("L")  # **转换为灰度模式，确保单通道**

        # **同步几何变换**
        img_image, segment_image = self.transforms(img_image, segment_image)

        # **转换为 Tensor**
        img_image = self.to_tensor(img_image)  # 变为 (3, H, W)
        segment_image = self.to_tensor(segment_image)  # **变为 (1, H, W)，避免通道不匹配**
        return img_image, segment_image
