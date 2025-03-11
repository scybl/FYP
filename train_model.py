import os
import torch
from torch import no_grad
from torchvision.utils import save_image

from LoadData.data import get_dataset
from LoadData.utils import load_config
from LossFunction.LearningRate import PolyWarmupScheduler
from LossFunction.LossChoose import LossFunctionHub
from model_defination.model_loader import load_model
from torch.optim import AdamW

class Trainer:
    def __init__(self, config_path, model_name, dataset_name):
        self.config = load_config(config_path)
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        self.class_num = self.config["datasets"][dataset_name]["class_num"]
        self.model_name = model_name

        self.train_loader = get_dataset(self.config, dataset_name, 'train')

        self.net = load_model(self.config, 'train', model_name, dataset_name).to(self.device)

        if self.class_num == 1:
            loss_hub = LossFunctionHub(loss_name="dice_ce", include_background=False, to_onehot_y=False, softmax=False,sigmoid=True) # 单分类
        else:
            loss_hub = LossFunctionHub(loss_name="dice_ce", include_background=True, to_onehot_y=False, softmax=True, sigmoid=False) # 多分类


        self.loss_fn = loss_hub.get_loss_function()

        self.opt = AdamW(self.net.parameters(), lr=self.config["setting"]['min_lr'], betas=(0.99, 0.95)) # AdamW 比 Adam 更适合现代深度学习任务，因为：
        self.scheduler = PolyWarmupScheduler(
            optimizer=self.opt,
            warmup_epochs=self.config["setting"]['warmup_epochs'],
            total_epochs=self.config["setting"]['epochs'],
            initial_lr=self.config["setting"]['max_lr'],
            power=0.9,
            eta_min=self.config["setting"]['min_lr']
        )
        self.dataset_name = dataset_name
        self.save_model_path = os.path.join(self.config['model']["save_path"], model_name)
        self.loss_log_path = os.path.join(self.config['model']['save_path'], f"train_loss_log_{model_name}_{self.dataset_name}.csv")
        self._init_log_file()

    def _init_log_file(self):
        with open(self.loss_log_path, "w") as f:
            f.write("epoch,step,train_loss\n")

    # def val(self):
    #     self.net.eval()
    #     with torch.no_grad():
    #         for i, (image, segment_image) in enumerate(self.data_loader): # self.val_loader
    #             image, segment_image = image.to(self.device), segment_image.to(self.device)
    #             out_image = self.net(image)
    #
    #             test_loss = self.loss_fn(out_image, segment_image)
    #             total += test_loss
    #             ##############################################################################################
    #             # 保存图像，用于可视化
    #             _image = image[0]
    #             _segment_image = segment_image[0]
    #             _out_image = out_image[0]
    #
    #             _segment_image = _segment_image.repeat(3, 1, 1)  # 重复 3 次通道，大小变为 [3, 256, 256]
    #             _out_image = _out_image.repeat(3, 1, 1)  # 重复 3 次通道，大小变为 [3, 256, 256]
    #
    #             img = torch.stack([_image, _segment_image, _out_image], dim=0)
    #             save_image(img, os.path.join(self.config['save_image_path'],
    #                                          f"{self.config['model']['name']}_{self.dataset_name}_{i}.png"))
    #
    #         return total/batch_size



    def train(self):
        epochs = 1
        best_dice = 0

        while epochs <= self.config["setting"]['epochs']:

            for i, (image, segment_image) in enumerate(self.train_loader):
                image, segment_image = image.to(self.device), segment_image.to(self.device)
                out_image = self.net(image)
                # print(f'image的大小为: f{image.size()}')
                # print(f'mask的大小为: f{segment_image.size()}')
                # print(f'out_img的大小为: f{out_image.size()}')

                train_loss = self.loss_fn(out_image, segment_image)

                self.opt.zero_grad()
                train_loss.backward()
                self.opt.step()

                # 保存日志
                with open(self.loss_log_path, "a") as f:
                    f.write(f"{epochs},{i},{train_loss.item():.6f}\n")

##############################################################################################
                # 保存图像，用于可视化
                _image = image[0]
                _segment_image = segment_image[0]
                _out_image = out_image[0]

                _segment_image = _segment_image.repeat(3, 1, 1)  # 重复 3 次通道，大小变为 [3, 256, 256]
                _out_image = _out_image.repeat(3, 1, 1)  # 重复 3 次通道，大小变为 [3, 256, 256]

                img = torch.stack([_image, _segment_image, _out_image], dim=0)
                save_image(img, os.path.join(self.config['save_image_path'],f"{self.model_name}_{self.dataset_name}_{i}.png"))
###############################################################################################

                # 打印训练信息
                current_lr = self.opt.param_groups[0]['lr']
                assert current_lr == self.scheduler.get_lr(), f"不相等,检查问题current_lr:{current_lr}, scheduler_lr:{self.scheduler.get_lr()}"
                print(f"Epoch {epochs} --- Step {i} --- Loss: {train_loss.item():.6f} --- LR: {current_lr:.6f}")


            # # call self.val
            # if best IS BETTER：
            # reroll the model to the best
            # else save the model and update best


            # 每个epoch保存一个模型
            torch.save(self.net.state_dict(), f"{self.save_model_path}_{self.dataset_name}_{epochs}.pth")

            # 更新学习率
            self.scheduler.step()
            epochs += 1

# 运行训练
if __name__ == "__main__":
    model_list = [
        "bnet",
        'unet',
        "bnet34",
    ]

    dataset_list = [
        'kvasir',
        'clinicdb',
        'isic2018',
        'sunapse'
    ]

    train_config_path = 'configs/config_train.yaml'
    for model_name in model_list:
        for dataset_name in dataset_list:
            print(dataset_name)
            print(model_name)

            trainer = Trainer(train_config_path,model_name=model_name,dataset_name=dataset_name)
            trainer.train()