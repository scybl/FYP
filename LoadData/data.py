from LoadData.ClinicDB_Dataset import ClinicDB_Dataset
from LoadData.ISIC2018_Dataset import ISIC2018_DataSet
from torch.utils.data import DataLoader

from LoadData.KvasirSEG_Dataset import KvasirSEG_Dataset
from LoadData.Synapse_Dataset import Synapse_Dataset


class LabelProcessor:
    """
    标签预处理器：将标签转换为指定的通道数，确保标签格式符合网络需求
    """
    def __init__(self, class_num=1):
        self.class_num = class_num

    def __call__(self, label):
        if self.class_num == 1:
            # 二分类任务，保留单通道
            if label.ndim == 3 and label.shape[0] > 1:
                label = label[0:1, :, :]  # 转为单通道
            elif label.ndim == 2:
                label = label.unsqueeze(0)  # 添加通道维度
            # 二值化标签，确保值在 0 和 1 之间
            label = (label > 0.5).float()
        else:
            # 多分类任务，假设标签已经是 one-hot 编码或分类索引形式
            # 如果是分类索引形式，直接返回
            if label.ndim == 3 and label.shape[0] != self.class_num:
                raise ValueError(f"Expected label channels {self.class_num}, but got {label.shape[0]}")
        return label


def get_dataset(config, mode):
    """
    通用数据加载器函数，用于获取训练或测试数据加载器。

    :param config: 配置字典，包含数据集和加载器的配置信息。
    :param mode: 数据模式，"train" 或 "test"，决定加载训练或测试数据。
    :return: 数据加载器 (DataLoader) 对象。
    """
    if mode not in ["train", "test"]:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'train' or 'test'.")

    # 选择不同模式下的 dataset_name
    dataset_name = config["setting"]["dataset_name"] if mode == "train" else config["setting"]["dataset_name"]

    # 如果数据集是 ISIC2018，加载特定数据集
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

    # 初始化数据集
    dataset = dataset_class(dataset_config)

    print(f"{dataset.__len__()}")

    data_all = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # 返回数据加载器
    return data_all
