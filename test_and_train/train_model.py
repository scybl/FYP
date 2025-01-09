import os
from torch import optim
from torchvision.utils import save_image
from LoadData.data import get_train_dataset
from LoadData.utils import load_config
import torch
import torch.nn as nn

from model_defination.model_loader import load_model_train
from test_and_train.cosineannealingLR import CosineAnnealingLR


def print_tensor_size(name, tensor):
    if tensor is None:
        print(f"{name}: Tensor is None")
    else:
        print(f"{name}: Size: {tensor.size()}, Device: {tensor.device}")


# load the config file
CONFIG_NAME = "config.yaml"
CONFIG_PATH = os.path.join("configs/", CONFIG_NAME)
config = load_config(CONFIG_PATH)

device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_config = config["train_setting"]
    net = load_model_train(config)
    opt = optim.Adam(net.parameters(), lr=train_config['lr'])
    loss_fn = nn.BCEWithLogitsLoss()

    # load data
    data_loader = get_train_dataset(config)
    save_model_path = os.path.join(config["model_path"], config["model_name"])

    # 使用余弦退火调度器
    lr_scheduler = CosineAnnealingLR(
        lr=train_config['lr'],  # 初始学习率，从配置文件中读取
        t_max=train_config['t_max'],  # 设置余弦退火周期，单位为 epoch 数，通常为训练的总 epoch 数或其倍数
        steps_per_epoch=len(data_loader),  # 每个 epoch 的步骤数，即每轮迭代的批次数量，通过数据集的长度获取
        max_epoch=train_config['epochs'],  # 总训练 epoch 数，从配置文件中读取
        warmup_epochs=train_config['warmup_epochs'],  # 预热的 epoch 数，在该阶段内学习率将线性增大到初始学习率
        eta_min=train_config['eta_min']  # 最小学习率，用于余弦退火过程的最终学习率下限
    )
    learning_rates = lr_scheduler.get_lr()  # 获取整个训练过程中的学习率数组
    # 打开一个文件用于保存每个step的损失
    loss_log_path = os.path.join(config['model_path'], ('train_loss_log_' + f"{config['model_name']}" + ".csv"))
    with open(loss_log_path, "w") as f:
        f.write("epoch,step,train_loss\n")  # 写入CSV文件的表头

    epochs = 1
    t = 1
    while epochs <= train_config['epochs']:
        for i, (image, segment_image) in enumerate(data_loader):
            # image is the original pic
            # segment image is the mark pic
            image, segment_image = image.to(device), segment_image.to(device)

            # 动态更新学习率
            current_lr = learning_rates[(epochs - 1) * len(data_loader) + i]
            for param_group in opt.param_groups:
                param_group['lr'] = current_lr

            # 前向传播、损失计算、反向传播和优化步骤
            out_image = net(image)  # 将输入图像通过网络，获得预测输出
            train_loss = loss_fn(out_image, segment_image)  # 计算训练损失
            opt.zero_grad()  # 清空优化器中的梯度信息
            train_loss.backward()  # 反向传播计算梯度
            opt.step()  # 更新模型权重参数

            # 保存模型和日志
            if t % train_config["save_interval"] == 0:
                torch.save(net.state_dict(), save_model_path + f"_{str(t / train_config['save_interval'])}.pth")
                # 将当前step的损失保存到日志文件
                with open(loss_log_path, "a") as f:
                    f.write(f'{epochs},{i},{train_loss.item():.6f}\n')

            # 保存图像，用于可视化
            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            _segment_image = _segment_image.repeat(3, 1, 1)  # 重复 3 次通道，大小变为 [3, 256, 256]
            _out_image = _out_image.repeat(3, 1, 1)  # 重复 3 次通道，大小变为 [3, 256, 256]

            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f"{config['save_image_path']}/{i}.png")

            # 打印训练信息，包括当前的 epoch，当前步骤索引和当前的训练损失及学习率
            print(
                f"Epoch {epochs} --- Step: {i} --- Train Loss: {train_loss.item()} --- Learning Rate: {current_lr:.6f}")
            t += 1
        epochs += 1
