import os
from builtins import print
from torch.utils.data import DataLoader
from LoadData.ISIC2018 import ISIC2018_DataSet, transform_image, get_transforms, AugmentedDataset


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
    if mode == "train":
        dataset_name = config["train_setting"]["dataset_name"]
        dataset_config = config["datasets"][dataset_name]
        batch_size = config["data_loader"]["batch_size"]
        shuffle = config["data_loader"]["shuffle"]
        num_workers = config["data_loader"]["num_workers"]
        augmentations = config["datasets"][dataset_name]["augmentations"]
    elif mode == "test":
        dataset_name = config["test_setting"]["dataset_name"]
        dataset_config = config["datasets"][dataset_name]
        batch_size = config.get("batch_size", 1)  # 默认值为1，防止遗漏配置
        shuffle = config.get("shuffle", False)  # 测试集一般不打乱数据
        num_workers = config.get("num_workers", 16)  # 默认值为4
        augmentations = []
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'train' or 'test'.")

    print(f"Loading {mode} dataset: {dataset_name}")

    # 获取数据集的 class_num
    class_num = dataset_config.get("class_num", 1)

    # 初始化数据集
    dataset = ISIC2018_DataSet(
        dataset_config,
        _transform_image=transform_image,
        transform_label=LabelProcessor(class_num=class_num),
        class_num=class_num
    )

    # 这是基础变换，后面还要加新的变换方式
    transform = get_transforms(augmentations)

    dataset = AugmentedDataset(base_dataset=dataset, transform=transform)

    # 返回数据加载器
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)
