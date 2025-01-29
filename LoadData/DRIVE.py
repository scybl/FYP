import os
from torch.utils.data import Dataset

from LoadData.assessment import transform_image, geometric, keep_image_size_open


class DRIVE_Dataset(Dataset):
    """
    DRIVE 数据集加载器。
    """

    def __init__(self, config, augmentations, transform_label=None):
        self.config = config
        self.image_names = os.listdir(os.path.join(self.config["dataset_path"], self.config["img"]))
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.augmentations = geometric(augmentations)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.config["dataset_path"], self.config["img"], image_name)

        mask_name = image_name.replace(self.config["img_suffix"], self.config["mask_suffix"])
        mask_path = os.path.join(self.config["dataset_path"], self.config["mask"], mask_name)

        image = keep_image_size_open(image_path, (self.config["size"], self.config["size"]))
        mask = keep_image_size_open(mask_path, (self.config["size"], self.config["size"]))

        if self.augmentations:
            image, mask = self.augmentations(image, mask)

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_label:
            mask = self.transform_label(transform_image(mask))

        return image, mask
