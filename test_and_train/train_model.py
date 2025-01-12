import os
from torch import optim
from torchvision.utils import save_image
from LoadData.data import get_dataset
from LoadData.utils import load_config
import torch
import torch.nn as nn

from model_defination.model_loader import load_model
from test_and_train.cosineannealingLR import CosineAnnealingLR


def print_tensor_size(name, tensor):
    if tensor is None:
        print(f"{name}: Tensor is None")
    else:
        print(f"{name}: Size: {tensor.size()}, Device: {tensor.device}")


# load the config file
CONFIG_NAME = "config_train.yaml"
CONFIG_PATH = os.path.join("configs/", CONFIG_NAME)
config = load_config(CONFIG_PATH)

device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_config = config["train_setting"]
    net = load_model(config, 'train')
    opt = optim.Adam(net.parameters(), lr=train_config['lr'])
    loss_fn = nn.BCEWithLogitsLoss()

    # load data
    data_loader = get_dataset(config, 'train')
    save_model_path = os.path.join(config['model']["save_path"], config["model"]['name'])

    # 使用余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=opt,
        T_max=train_config['t_max'],  # 设置余弦退火周期
        eta_min=train_config['eta_min']  # 最小学习率
    )

    # 日志路径
    loss_log_path = os.path.join(config['model']['save_path'], ('train_loss_log_' + f"{config['model']['name']}" + ".csv"))
    with open(loss_log_path, "w") as f:
        f.write("epoch,step,train_loss\n")  # 写入CSV文件的表头

    epochs = 1
    t = 1
    while epochs <= train_config['epochs']:
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            # 前向传播、损失计算、反向传播和优化步骤
            out_image = net(image)  # 网络预测
            train_loss = loss_fn(out_image, segment_image)  # 计算损失
            opt.zero_grad()  # 清空梯度
            train_loss.backward()  # 反向传播
            opt.step()  # 更新模型权重

            # 更新学习率
            scheduler.step()

            # 保存模型和日志
            if t % train_config["save_interval"] == 0:
                torch.save(net.state_dict(), save_model_path + f'_{str(t / train_config["save_interval"])}.pth')
                with open(loss_log_path, "a") as f:
                    f.write(f'{epochs},{i},{train_loss.item():.6f}\n')

            # 打印训练信息
            current_lr = opt.param_groups[0]['lr']
            print(f"Epoch {epochs} --- Step: {i} --- Train Loss: {train_loss.item()} --- Learning Rate: {current_lr:.6f}")
            t += 1
        epochs += 1
