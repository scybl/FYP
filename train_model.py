import os
import torch
from torchvision.utils import save_image

from Evaluate.evaluate import dice
from LoadData.data import get_dataset
from LoadData.utils import load_config
from Evaluate.LearningRate import PolyWarmupScheduler
from Evaluate.LossChoose import LossFunctionHub
from model_defination.model_loader import load_model
from torch.optim import AdamW

from fvcore.nn import FlopCountAnalysis, parameter_count_table



class Trainer:
    def __init__(self, config_path, model_name, dataset_name):
        self.config = load_config(config_path)
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        self.class_num = self.config["datasets"][dataset_name]["class_num"]
        self.model_name = model_name

        self.dataset_name = dataset_name
        self.train_dataset = get_dataset(self.config, self.dataset_name, 'train')
        self.val_dataset = get_dataset(self.config, self.dataset_name, 'val')  # 这里写入验证数据集

        self.net = load_model(self.config, model_name, dataset_name).to(self.device)

        if self.class_num == 1:
            loss_hub = LossFunctionHub(loss_name="dice_ce", include_background=False, to_onehot_y=False, softmax=False,
                                       sigmoid=True)  # 单分类
        else:
            loss_hub = LossFunctionHub(loss_name="dice_ce", include_background=True, to_onehot_y=False, softmax=True,
                                       sigmoid=False)  # 多分类

        self.loss_fn = loss_hub.get_loss_function()

        self.opt = AdamW(self.net.parameters(), lr=self.config["setting"]['min_lr'],
                         betas=(0.99, 0.95))  # AdamW 比 Adam 更适合现代深度学习任务，因为：
        self.scheduler = PolyWarmupScheduler(
            optimizer=self.opt,
            warmup_epochs=self.config["setting"]['warmup_epochs'],
            total_epochs=self.config["setting"]['epochs'],
            initial_lr=self.config["setting"]['max_lr'],
            power=0.9,
            eta_min=self.config["setting"]['min_lr']
        )
        self.save_model_path = os.path.join(self.config['model']["save_path"], model_name)
        self.loss_log_path = os.path.join(self.config['model']['save_path'],
                                          f"log_{model_name}_{self.dataset_name}.csv")
        self._init_log_file()

    def _init_log_file(self):
        file_exists = os.path.exists(self.loss_log_path)
        mode = "a" if file_exists else "w"
        with open(self.loss_log_path, mode) as f:
            if not file_exists:
                f.write("epoch,step,train_loss\n")

    def val(self) -> float:
        self.net.eval()  # 设置评估模式
        num_batches = 0
        dice_scores = []

        with torch.no_grad():
            for i, (image, segment_image) in enumerate(self.val_dataset):  # 或者使用 self.val_loader
                image, segment_image = image.to(self.device), segment_image.to(self.device)
                out_image = torch.sigmoid(self.net(image)) # 二分类激活

                # 计算 Dice 指标
                dice_score = dice(pred=out_image, target=segment_image)
                dice_scores.append(dice_score)

                num_batches += 1

                ####################################################################################
                # 保存图像，用于可视化
                _image = image[0]
                _segment_image = segment_image[0]
                _out_image = out_image[0]

                # 将单通道重复 3 次以便可视化
                _segment_image = _segment_image.repeat(3, 1, 1)
                _out_image = _out_image.repeat(3, 1, 1)

                # 堆叠图像：原图、标签图、输出图
                img = torch.stack([_image, _segment_image, _out_image], dim=0)
                save_path = os.path.join(self.config['save_image_path'],
                                         f"{self.model_name}_{self.dataset_name}_{i}.png")
                save_image(img, save_path)
                ####################################################################################

        # 计算平均 Dice
        avg_dice = sum(dice_scores) / num_batches
        print(f"Validation Dice: {avg_dice:.6f}")
        return avg_dice

    def train(self):
        epochs = 1
        best_val = self.val()

        while epochs <= self.config["setting"]['epochs']:
            self.net.train()  # 确保模型处于训练模式
            for i, (image, segment_image) in enumerate(self.train_dataset):
                self.opt.zero_grad()
                image, segment_image = image.to(self.device), segment_image.to(self.device)
                out_image = self.net(image)
                # print(f'image的大小为: f{image.size()}')
                # print(f'mask的大小为: f{segment_image.size()}')
                # print(f'out_img的大小为: f{out_image.size()}')

                train_loss = self.loss_fn(out_image, segment_image)

                train_loss.backward()
                self.opt.step()

                # 保存训练日志
                with open(self.loss_log_path, "a") as f:
                    f.write(f"{epochs},{i},{train_loss.item():.6f}\n")

                current_lr = self.opt.param_groups[0]['lr']
                # 注意：scheduler.get_lr() 可能返回列表，这里需要根据实际情况调整断言
                print(f"Epoch {epochs} --- Step {i} --- Loss: {train_loss.item():.6f} --- LR: {current_lr:.6f}")

            # 每个 epoch 后调用验证函数
            val_dice = self.val()

            # 保存最优模型逻辑（示例：当验证 loss 更低时保存）
            if val_dice > best_val:
                best_val = val_dice
                torch.save(self.net.state_dict(), f"{self.save_model_path}_{self.dataset_name}_best.pth")
                print(f"Epoch {epochs}: 找到更优模型，保存模型。")


            # 更新学习率
            self.scheduler.step()
            epochs += 1

    def analyze(self, input_tensor_size):
        """
        根据输入张量尺寸计算模型的 FLOPs 和参数数量，并打印结果。

        参数:
        input_tensor_size (tuple): 输入张量尺寸，格式为 (Channels, Height, Width)
        """
        # 构造一个与模型输入匹配的随机张量 (Batch size 默认为 1)
        input_tensor = torch.randn(1, *input_tensor_size).to(self.device)

        # 使用 FlopCountAnalysis 计算 FLOPs
        flops = FlopCountAnalysis(self.net, input_tensor)
        print("Total FLOPs:", flops.total())

        # 输出模型参数数量统计
        print("\nParameters:")
        print(parameter_count_table(self.net))


# 运行训练
if __name__ == "__main__":
    
    model_hub = [
        # "duck",
        # "unetpp",
        # "bnet",
        # 'unet',
        # "bnet34",
        # 'unext',
        'dga',
        'pham'
    ]
    dataset_hub = [
        'kvasir',
        # 'clinicdb',
        # 'isic2018',
        # 'synapse'
    ]

    train_config_path = 'configs/config.yaml'
    for model_name in model_hub:
        for dataset_name in dataset_hub:

            print('-----------------')
            trainer = Trainer(train_config_path, model_name=model_name, dataset_name=dataset_name)
            trainer.analyze((3,224,224))
            # trainer.train()

