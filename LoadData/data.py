from LoadData.ClinicDB_Dataset import ClinicDB_Dataset
from LoadData.ISIC2018_Dataset import ISIC2018_DataSet
from torch.utils.data import DataLoader

from LoadData.KvasirSEG_Dataset import KvasirSEG_Dataset
from LoadData.Synapse_Dataset import Synapse_Dataset

import logging


class LabelProcessor:
    """
        Label preprocessor: Converts the labels to the specified number of channels,
        ensuring that the label format meets the network requirements
    """

    def __init__(self, class_num=1):
        self.class_num = class_num

    def __call__(self, label):
        if self.class_num == 1:
            # binary class
            if label.ndim == 3 and label.shape[0] > 1:
                label = label[0:1, :, :]
            elif label.ndim == 2:
                label = label.unsqueeze(0)
            label = (label > 0.5).float()
        else:
            # Multiclass classification
            if label.ndim == 3 and label.shape[0] != self.class_num:
                raise ValueError(f"Expected label channels {self.class_num}, but got {label.shape[0]}")
        return label


def get_dataset(config, dataset_name, mode):
    """
        General data loader function to get the training or testing data loader.

        :param config: Configuration dictionary containing dataset and loader configuration information.
        :param mode: Data mode, "train" or "test", determines whether to load training or testing data.
        :return: DataLoader object.
    """
    if mode not in ["train", "test", "val"]:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'train' or 'test'. or val")

    if dataset_name.lower() == "isic2018":
        dataset_class = ISIC2018_DataSet
    elif dataset_name.lower() == 'kvasir':
        dataset_class = KvasirSEG_Dataset
    elif dataset_name.lower() == 'clinicdb':
        dataset_class = ClinicDB_Dataset
    elif dataset_name.lower() == 'synapse':
        dataset_class = Synapse_Dataset
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    dataset_config = config["datasets"][dataset_name]
    batch_size = config["data_loader"]["batch_size"]
    shuffle = config["data_loader"]["shuffle"]
    num_workers = config["data_loader"]["num_workers"]

    print(f"Loading {mode} dataset: {dataset_name}, data augmentations has been loaded")

    # init dataset
    dataset = dataset_class(dataset_config, mode)

    data_all = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # 返回数据加载器
    return data_all
