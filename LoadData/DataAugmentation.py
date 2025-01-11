from torchvision import transforms
from PIL import Image
import random


class DataAugmentation:
    """
    A class to dynamically apply data augmentation methods based on input list.
    """

    def __init__(self, augmentations=None, image_size=(384, 384)):
        """
        Initialize the CustomAugmentation class.

        :param augmentations: List of augmentations to apply, e.g., ["反转", "旋转"].
        :param image_size: Target size for resizing the images.
        """
        self.image_size = image_size
        self.augmentations = augmentations if augmentations else []

        self.transform_list = []
        self.gt_transform_list = []

        for aug in self.augmentations:
            if aug == "反转":
                self.transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
                self.transform_list.append(transforms.RandomVerticalFlip(p=0.5))
                self.gt_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
                self.gt_transform_list.append(transforms.RandomVerticalFlip(p=0.5))
            elif aug == "旋转":
                self.transform_list.append(transforms.RandomRotation(90))
                self.gt_transform_list.append(transforms.RandomRotation(90))
            # Add more augmentations as needed
            elif aug == "裁剪":
                self.transform_list.append(transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)))
                self.gt_transform_list.append(transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)))

        # Add resizing, tensor conversion, and normalization to the pipeline
        self.transform_list.append(transforms.Resize(self.image_size))
        self.transform_list.append(transforms.ToTensor())
        self.transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        self.gt_transform_list.append(transforms.Resize(self.image_size))
        self.gt_transform_list.append(transforms.ToTensor())

        # Create composed transforms
        self.img_transform = transforms.Compose(self.transform_list)
        self.gt_transform = transforms.Compose(self.gt_transform_list)

    def apply(self, image, gt):
        """
        Apply the augmentations to both the image and ground truth.

        :param image: Input image (PIL Image).
        :param gt: Ground truth (PIL Image).
        :return: Augmented image and ground truth.
        """
        # Ensure reproducibility by syncing random seeds
        seed = random.randint(0, 2147483647)
        random.seed(seed)
        image = self.img_transform(image)

        random.seed(seed)
        gt = self.gt_transform(gt)

        return image, gt

