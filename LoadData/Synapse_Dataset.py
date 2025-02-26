import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from LoadData.assistance import build_transforms

"""
这个是synapse的数据集加载的替换
"""

class Synapse_Dataset(Dataset):
    def __init__(self, config, split='train'):
        self.split = split
        self.list_dir = config["list_dir"]
        self.sample_list = open(os.path.join(self.list_dir, self.split + '.txt')).readlines()
        self.data_dir = config["root_path"]
        # **使用 SynchronizedTransform 进行同步数据增强**
        self.transforms = build_transforms(config['augmentations'])  # 保留 transform_label

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx].strip('\n') # 获取样本名称

        if self.split == "train":
            data_path = os.path.join(self.data_dir, case_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            filepath = os.path.join(self.data_dir, f"{case_name}.npy.h5")
            with h5py.File(filepath, 'r') as data: # 以只读模式打开
                image, label = np.array(data['image']), np.array(data['label'])


        # 将 NumPy 数组转换为 Tensor，并且在第一个维度添加一个维度 (1, 512, 512)
        image = torch.tensor(image).unsqueeze(0)  # 变成形状 [1, 512, 512]
        label = torch.tensor(label).unsqueeze(0)  # 变成形状 [1, 512, 512]

        image, label = self.transforms(image, label)
        return image, label