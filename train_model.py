import os

import torchvision.utils as vutils

from PIL import Image
import torch
import numpy as np
from LoadData.data import get_dataset
from LoadData.utils import load_config
from model_defination.model_loader import load_model
from torch import optim
import torch.nn as nn

# load the config file
CONFIG_NAME = "config_train.yaml"
CONFIG_PATH = os.path.join("configs/", CONFIG_NAME)
config = load_config(CONFIG_PATH)

device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    train_config = config["train_setting"]
    net = load_model(config, 'train')
    opt = optim.Adam(net.parameters(), lr=train_config['lr'])
    loss_fn = nn.CrossEntropyLoss()  # 换成cross entropy损失，保证多目标可以使用

    # 加载数据
    data_loader = get_dataset(config, 'train')
    save_model_path = os.path.join(config['model']["save_path"], config["model"]['name'])

    # 余弦退火调度器 (更新学习率的周期 T_max 设为总 epoch 数)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=opt,
        T_max=train_config['epochs'],  # 以 epoch 为单位
        eta_min=train_config['eta_min']  # 最小学习率
    )

    # 日志路径
    loss_log_path = os.path.join(config['model']['save_path'], f"train_loss_log_{config['model']['name']}.csv")
    with open(loss_log_path, "w") as f:
        f.write("epoch,step,train_loss\n")  # 记录表头

    epochs = 1
    t = 1
    while epochs <= train_config['epochs']:
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            # 确保 segment_image 是 LongTensor 且 shape 正确
            segment_image = segment_image.squeeze(1).long()

            # 前向传播
            out_image = net(image)  # 不能加 softmax

            # 打印调试信息
            print(f"segment_image dtype: {segment_image.dtype}, unique values: {segment_image.unique()}")
            print(f"out_image shape: {out_image.shape}, segment_image shape: {segment_image.shape}")

            # 计算损失
            train_loss = loss_fn(out_image, segment_image)

            # 反向传播和优化
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            # 保存日志 (每个 batch 记录一次)
            with open(loss_log_path, "a") as f:
                f.write(f"{epochs},{i},{train_loss.item():.6f}\n")

            # 打印训练信息
            current_lr = opt.param_groups[0]['lr']
            print(f"Epoch {epochs} --- Step {i} --- Loss: {train_loss.item():.6f} --- LR: {current_lr:.6f}")

            # 保存模型 (按 save_interval)
            if t % train_config["save_interval"] == 0:
                torch.save(net.state_dict(), f"{save_model_path}_{t // train_config['save_interval']}.pth")

            t += 1

        # **在 epoch 级更新学习率**
        scheduler.step()
        epochs += 1


