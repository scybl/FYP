import os
import torch
from monai.networks import one_hot
from LoadData.data import get_dataset
from LoadData.utils import load_config
from LossFunction.LearningRate import PolyWarmupScheduler
from LossFunction.LossChoose import LossFunctionHub
from model_defination.model_loader import load_model
from torch.optim import AdamW


def print_tensor_size(name, tensor):
    if tensor is None:
        print(f"{name}: Tensor is None")
    else:
        print(f"{name}: Size: {tensor.size()}, Device: {tensor.device}")

#锁定参数

# load the config file
CONFIG_NAME = "config_train.yaml"
CONFIG_PATH = os.path.join("configs/", CONFIG_NAME)
config = load_config(CONFIG_PATH)

class_num = config["datasets"][config["setting"]["dataset_name"]]["class_num"]

device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

# TODO： 作一下encoder的部分，调用一个预训练模型
# # encoder: 换ConvNeXt V2, load 参数

# TODO：看一下transunet的数据集加载部分，用这个替换一下
# TransUnet Snapse https://github.com/Beckschen/TransUNet/blob/main/datasets/dataset_synapse.py
# # 30 组,  18train.  12 测试
if __name__ == "__main__":
    # 加载数据
    data_loader = get_dataset(config, 'train')
    save_model_path = os.path.join(config['model']["save_path"], config["model"]['name'])

    # 加载模型
    train_config = config["setting"]
    net = load_model(config, 'train')

    # opt = optim.Adam(net.parameters(), lr=train_config['lr'])
    opt = AdamW(net.parameters(), lr=train_config['lr'], betas=(0.99, 0.95))
    # 使用adamw优化器这个效果更好 AdamW, beta = (0.99, 0.95)
    # • AdamW 比 Adam 更适合现代深度学习任务，因为：
    # • 它使用 权重衰减（weight decay），可以 避免过拟合。
    # • 适用于 Transformer 和 CNN 任务。
    # • beta 参数 (0.99, 0.95) 更适合大规模训练。


    # MONAI https://docs.monai.io/en/stable/losses.html 用这个包
    loss_hub = LossFunctionHub(loss_name="dice_ce",include_background = True, to_onehot_y=False, softmax=True)
    loss_fn = loss_hub.get_loss_function()
    # loss_hub = LossFunctionHub(loss_name="cross_entropy", weight=None)
    # loss_fn = loss_hub.get_loss_function()


    # TODO: 更改一下学习率
    # Poly Strategy, H2Former，
    # warmup, 训练 K 轮之后，lr再降低， Spark: https://github.com/keyu-tian/SparK/blob/main/pretrain/utils/lr_control.py, warm-up 锁定，K轮次之后再降低lr
    scheduler = PolyWarmupScheduler(
        optimizer=opt,
        warmup_epochs=train_config['warmup_epochs'],
        total_epochs=train_config['epochs'],
        initial_lr=train_config['lr'],
        power=0.9,
        eta_min=train_config['eta_min']
    )

    # 日志路径
    loss_log_path = os.path.join(config['model']['save_path'], f"train_loss_log_{config['model']['name']}.csv")
    with open(loss_log_path, "w") as f:
        f.write("epoch,step,train_loss\n")  # 记录表头

    epochs = 1
    while epochs <= train_config['epochs']:

        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            # 前向传播
            out_image = net(image)

            # 看一下数据的size是否相同，如果相同则可以继续进行
            # print("image size: "+ f"{image.size()}")
            # print("out_image size: "+ f"{out_image.size()}")
            # print("mask image size:  "f"{segment_image.size()}")
            train_loss = loss_fn(out_image, segment_image)  # 所有的数据类型都是tensor float

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            # 保存日志 (每个 batch 记录一次)
            with open(loss_log_path, "a") as f:
                f.write(f"{epochs},{i},{train_loss.item():.6f}\n")

            # 打印训练信息
            current_lr = opt.param_groups[0]['lr']
            print(f"Epoch {epochs} --- Step {i} --- Loss: {train_loss.item():.6f} --- LR: {current_lr:.6f}")

        # 每个epoch保存一个模型
        torch.save(net.state_dict(), f"{save_model_path}_{epochs}.pth")

        # **在 epoch 级更新学习率**
        scheduler.step()
        epochs += 1


