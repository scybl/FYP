import os
from torch.utils.data import Dataset
from LoadData.assessment import load_image, transform_image, geometric_trans, color_trans


class ISIC2018_DataSet(Dataset):
    """
    自定义数据集，加载图像和对应的标签
    """

    def __init__(self, config, augmentations, transform_label=None, class_num=1):
        self.config = config
        self.mask_name = os.listdir(os.path.join(self.config["dataset_path"], self.config["mask"]))
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.class_num = class_num

        # **在初始化时解析数据增强**
        self.geometric_trans = geometric_trans(augmentations["geometric_transforms"])
        self.color_trans = color_trans(augmentations["color_transforms"])

    def __len__(self):
        return len(self.mask_name)

    def __getitem__(self, index):
        # 获取掩膜文件名和路径
        segment_name = self.mask_name[index]
        segment_path = os.path.join(self.config["dataset_path"], self.config['mask'], segment_name)

        # 根据掩膜文件名生成图像文件名
        image_name = segment_name.replace(self.config["seg_prefix"], self.config["img_prefix"]).replace(
            self.config["seg_suffix"], self.config["img_suffix"])
        image_path = os.path.join(self.config["dataset_path"], self.config["img"], image_name)

        # 加载图像和标签
        segment_image = load_image(segment_path, (self.config["size"], self.config["size"]))
        img_image = load_image(image_path, (self.config["size"], self.config["size"]))

        # 应用图像和标签的变换
        if self.transform_image:
            img_image = self.transform_image(img_image)
        if self.transform_label:
            segment_image = self.transform_label(transform_image(segment_image))

        return img_image, segment_image
