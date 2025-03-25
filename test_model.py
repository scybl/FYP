import os
import torch
from torchvision.utils import save_image

from LoadData.data import get_dataset
from LoadData.utils import load_config
from Evaluate.LossChoose import LossFunctionHub
from model_defination.model_loader import load_model

# import warnings
# warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")

class Tester:
    def __init__(self, config_path, _model_name, _dataset_name):
        # 加载配置
        self.config = load_config(config_path)
        # 设置设备
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        # 获取类别数
        self.class_num = self.config["datasets"][_dataset_name]["class_num"]
        self.model_name = _model_name
        self.dataset_name = _dataset_name

        # 加载测试数据集，这里假设get_dataset支持"test"模式
        self.test_dataset = get_dataset(self.config, self.dataset_name, 'test')

        # 加载模型并移动到设备，注意测试时可能需要特殊设置，例如 batch_size 调整等
        self.net = load_model(self.config, _model_name, _dataset_name).to(self.device)

        # 根据类别数选择损失函数
        if self.class_num == 1:
            loss_hub = LossFunctionHub(loss_name="dice_ce", include_background=False,
            to_onehot_y=False, softmax=False, sigmoid=True)
        else:
            loss_hub = LossFunctionHub(loss_name="dice_ce", include_background=True,
            to_onehot_y=False, softmax=True, sigmoid=False)
        self.loss_fn = loss_hub.get_loss_function()
        self.save_image_path = self.config['save_image_path']

    def test(self) -> float:
        self.net.eval() # 设置为评估模式
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, (image, segment_image) in enumerate(self.test_dataset):
                image, segment_image = image.to(self.device), segment_image.to(self.device)
                out_image = self.net(image)

                loss = self.loss_fn(out_image, segment_image)
                total_loss += loss.item()
                num_batches += 1

                ####################################################################################
                # 保存测试结果图像用于可视化
                _image = image[0]
                _segment_image = segment_image[0]
                _out_image = out_image[0]

                # 如果标签和输出为单通道，复制3通道方便可视化
                _segment_image = _segment_image.repeat(3, 1, 1)
                _out_image = _out_image.repeat(3, 1, 1)

                # 堆叠原图、标签图、预测图
                img = torch.stack([_image, _segment_image, _out_image], dim=0)
                save_path = os.path.join(self.save_image_path,
                f"{self.model_name}_{self.dataset_name}_test_{i}.png")
                save_image(img, save_path)
                ####################################################################################

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Test Loss: {1.0 - avg_loss:.6f}")
        return avg_loss

if __name__ == "__main__":
    model_hub = [
        # "duck",
        "unetpp",
        "bnet",
        'unet',
        "bnet34",
    ]
    dataset_hub = [
        'kvasir',
        'clinicdb',
        'isic2018',
        # 'synapse'
    ]

    test_config_path = 'configs/config.yaml' # 测试配置文件路径，可以与训练配置共用

    for model_name in model_hub:
        for dataset_name in dataset_hub:
            print("---------------")
            tester = Tester(test_config_path, _model_name=model_name, _dataset_name=dataset_name)
            tester.test()
