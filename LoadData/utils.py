import yaml
import random
import torch
import torchvision.transforms as transforms


class SynchronizedTransform:
    """
    Ensure that the image and mask undergo the same augmentation operations simultaneously
    """

    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, mask=None):
        seed = random.randint(0, 2 ** 32)  # random seed
        random.seed(seed)
        torch.manual_seed(seed)

        for transform in self.transforms:
            if isinstance(transform, transforms.ColorJitter):
                image = transform(image)
            else:
                image = transform(image)
                if mask is not None:
                    random.seed(seed)
                    torch.manual_seed(seed)
                    mask = transform(mask)

        return image, mask


def build_transforms(augmentations):
    transform_list = []

    if augmentations.get("geometric_transforms") is not None:
        for aug in augmentations.get("geometric_transforms"):
            transform_type = aug["type"]
            params = aug.get("params", {})

            if transform_type == "RandomHorizontalFlip":
                transform_list.append(transforms.RandomHorizontalFlip(p=params.get("p", 0.5)))
            elif transform_type == "RandomRotation":
                transform_list.append(transforms.RandomRotation(degrees=params.get("degree", 15)))
            elif transform_type == "Resize":
                transform_list.append(transforms.Resize(size=params.get("size", (256, 256))))
            else:
                raise ValueError(f"Unsupported geometric transformation: {transform_type}")

        color_params = {aug["type"]: aug["params"]["degree"] for aug in augmentations.get("color_transforms", [])}
        if color_params:
            transform_list.append(transforms.ColorJitter(
                brightness=color_params.get("brightness"),
                contrast=color_params.get("contrast"),
                saturation=color_params.get("saturation"),
                hue=color_params.get("hue")
            ))

    return SynchronizedTransform(transform_list)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
