import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

from LoadData.utils import build_transforms, get_split_paths, list_dataset_files


class KvasirSEG_Dataset(data.Dataset):

    def __init__(self, config, mode):
        self.config = config
        self.class_num = config["class_num"]
        self.dataset_path = self.config["dataset_path"]
        self.img_path, self.mask_path = get_split_paths(self.config, mode)

        self.img_list = list_dataset_files(self.img_path)
        self.mask_list = list_dataset_files(self.mask_path)

        self.transforms = build_transforms(config['augmentations'])

        self.to_tensor = transforms.ToTensor()
        self.transform_label = None

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_file_name = self.mask_list[index]

        mask_file = os.path.join(self.mask_path, mask_file_name)

        image_name = mask_file_name

        image_path = os.path.join(self.img_path, image_name)

        img_image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_file).convert("L")

        img_image, mask_image = self.transforms(img_image, mask_image)

        if self.transform_label:
            mask_image = self.transform_label(mask_image)

        img_image = self.to_tensor(img_image)  # (3, H, W)
        mask_image = self.to_tensor(mask_image)  # (1, H, W)
        return img_image, mask_image
