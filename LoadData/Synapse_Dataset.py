import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import torchvision.transforms as transforms

"""
这个是synapse的数据集加载的替换
"""

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample



class Synapse_Dataset(Dataset):
    def __init__(self, config, split='train'):
        self.split = split
        self.list_dir = config["list_dir"]
        self.sample_list = open(os.path.join(self.list_dir, self.split + '.txt')).readlines()
        self.data_dir = config["root_path"]
        self.to_tensor = transforms.ToTensor()

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


        return image, label