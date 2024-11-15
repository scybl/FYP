from torch.utils.data import Dataset
from LoadData.utils import keep_image_size_open
import os
from torchvision import transforms

from torch.utils.data import DataLoader

"""
这个方法是用来加载数据集，通过返回data数据来加载全部数据
输入项目地址，自动加载位于pwd+isic2018位置中的数据集
"""
# TODO:后续可以将isic分离出来用来加载别的项目数据集
transform = transforms.Compose([
    transforms.ToTensor()
])  # 定义一个图像变换 transform，把加载的图像转换为 PyTorch 中的 Tensor 格式，并将像素值归一化到 [0.0, 1.0] 范围，以便后续的神经网络训练或处理。


class MyDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.mask_name = os.listdir(os.path.join(self.config["dataset_path"], self.config["train_data_mask"]))

    def __len__(self):
        """
        :return: the length of the dataset
        """
        return len(self.mask_name)

    def __getitem__(self, index):
        segment_name = self.mask_name[index]  # seg_name = prefix + name + suffix
        segment_path = os.path.join(self.config["dataset_path"], self.config["train_data_mask"], segment_name)
        # replace the name by suffix and prefix, for ISIC2018 prefix is no need, so ignore prefix

        # 将mask文件名替换为相应的image文件名
        image_name = segment_name.replace(self.config["seg_prefix"], self.config["img_prefix"]).replace(
            self.config["seg_suffix"], self.config["img_suffix"])
        image_path = os.path.join(self.config["dataset_path"], self.config["train_data_img"], image_name)

        segment_image = keep_image_size_open(segment_path, (self.config["size"], self.config["size"]))
        img_image = keep_image_size_open(image_path, (self.config["size"], self.config["size"]))

        return transform(img_image), transform(segment_image)


def get_train_dataset(config):
    dataset_name = config["train_setting"]["train_dataset_name"]
    print(f"load the dataset is {dataset_name} 1")
    return DataLoader(MyDataset(config["train_setting"]["train_dataset"][dataset_name]),
                      batch_size=config['batch_size'],
                      shuffle=config['shuffle'],
                      num_workers=config['num_workers'])


def get_test_dataset(config):
    dataset_name = config["test_setting"]["test_dataset_name"]
    print(f"load the dataset is {dataset_name}")
    return DataLoader(MyDataset(config["test_setting"]["test_dataset"][dataset_name]),
                      batch_size=config['batch_size'],
                      shuffle=config['shuffle'],
                      num_workers=config['num_workers'])
