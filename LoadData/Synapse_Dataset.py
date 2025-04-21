import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from LoadData.utils import build_transforms

"""
Synapse dataset loader
"""


class Synapse_Dataset(Dataset):
    def __init__(self, config, split='train'):
        self.split = split
        self.list_dir = config["list_dir"]
        self.sample_list = open(os.path.join(self.list_dir, self.split + '.txt')).readlines()
        self.data_dir = config["root_path"]

        self.transforms = build_transforms(config['augmentations'])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx].strip('\n')

        if self.split == "train":
            data_path = os.path.join(self.data_dir, case_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            filepath = os.path.join(self.data_dir, f"{case_name}.npy.h5")
            with h5py.File(filepath, 'r') as data:
                image, label = np.array(data['image']), np.array(data['label'])

        image = torch.tensor(image).unsqueeze(0)  # [1, 512, 512]
        label = torch.tensor(label).unsqueeze(0)  # [1, 512, 512]

        image, label = self.transforms(image, label)
        return image, label
