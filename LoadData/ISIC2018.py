import os
from torch.utils.data import Dataset
from LoadData.assessment import build_transforms
from PIL import Image
import torchvision.transforms as transforms


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
        print(f"DEBUG: image shape = {img_image.shape}, mask shape = {segment_image.shape}")
        return img_image, segment_image
