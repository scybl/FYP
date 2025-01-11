import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# 定义图像和标签的变换
transform_image = transforms.Compose([
    transforms.ToTensor()
])  # 图像转换为张量


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


# 保持图像大小的打开方法
def keep_image_size_open(file_path, size):
    image = Image.open(file_path).convert('RGB')  # 强制为 RGB 格式
    image = image.resize(size, Image.BILINEAR)
    return image


class MyDataset(Dataset):
    """
    自定义数据集，加载图像和对应的标签
    """

    def __init__(self, config, _transform_image=None, transform_label=None, class_num=1):
        self.config = config
        self.mask_name = os.listdir(os.path.join(self.config["dataset_path"], self.config["mask"]))
        self.transform_image = _transform_image
        self.transform_label = transform_label
        self.class_num = class_num

    def __len__(self):
        return len(self.mask_name)

    def __getitem__(self, index):
        # 获取掩膜文件名和路径
        segment_name = self.mask_name[index]
        segment_path = os.path.join(self.config["dataset_path"], self.config['mask'], segment_name)

        # 根据掩膜文件名生成图像文件名
        image_name = segment_name.replace(self.config["seg_prefix"], self.config["img_prefix"]).replace(
            self.config["seg_suffix"], self.config["img_suffix"])
        image_path = os.path.join(self.config["dataset_path"], self.config["img"], image_name)

        # 加载图像和标签
        segment_image = keep_image_size_open(segment_path, (self.config["size"], self.config["size"]))
        img_image = keep_image_size_open(image_path, (self.config["size"], self.config["size"]))

        # 应用图像和标签的变换
        if self.transform_image:
            img_image = self.transform_image(img_image)
        if self.transform_label:
            segment_image = self.transform_label(transform_image(segment_image))

        return img_image, segment_image


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

    elif mode == "test":
        dataset_name = config["test_setting"]["dataset_name"]
        dataset_config = config["datasets"][dataset_name]
        batch_size = config.get("batch_size", 1)  # 默认值为1，防止遗漏配置
        shuffle = config.get("shuffle", False)  # 测试集一般不打乱数据
        num_workers = config.get("num_workers", 4)  # 默认值为4

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'train' or 'test'.")

    print(f"Loading {mode} dataset: {dataset_name}")

    # 获取数据集的 class_num
    class_num = dataset_config.get("class_num", 1)

    # 初始化数据集
    dataset = MyDataset(
        dataset_config,
        _transform_image=transform_image,
        transform_label=LabelProcessor(class_num=class_num),
        class_num=class_num
    )

    # 返回数据加载器
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)
