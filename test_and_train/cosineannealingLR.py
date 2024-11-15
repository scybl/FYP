import math
import numpy as np

"""
余弦退火学习率调度器的实现，结合了线性预热（Linear Warm-Up）功能。
此调度器适用于深度学习模型训练中，通过余弦退火的方式逐步降低学习率，
使模型在训练后期更平稳地接近全局最优解。

余弦退火（Cosine Annealing）是一种动态调整学习率的策略，常用于深度学习模型的训练过程。
在每个 epoch 中，学习率以余弦函数的方式逐步减少，形成一个非线性的衰减曲线。
在训练的后期，学习率接近预设的最小值，通常可以让模型收敛得更稳定。
"""


# 定义线性预热学习率调度器类，用于在训练初期逐步增加学习率
class _LinearWarmUp:
    def __init__(self, lr, warmup_epochs, steps_per_epoch, warmup_init_lr=0):
        """
        初始化线性预热调度器。

        参数:
        - lr: 基础学习率，在预热阶段结束时达到的目标学习率。
        - warmup_epochs: 预热阶段的轮数。
        - steps_per_epoch: 每个 epoch 中的步骤数（即批次数）。
        - warmup_init_lr: 预热的初始学习率（默认值为 0）。
        """
        self.base_lr = lr  # 目标学习率
        self.warmup_init_lr = warmup_init_lr  # 初始学习率
        self.warmup_steps = int(warmup_epochs * steps_per_epoch)  # 总的预热步骤数

    def get_warmup_steps(self):
        """返回预热步骤总数"""
        return self.warmup_steps

    def get_lr(self, current_step):
        """
        根据当前步骤计算当前学习率。

        参数:
        - current_step: 当前的步骤（从 1 到 warmup_steps）

        返回:
        - 计算出的当前学习率
        """
        # 学习率增量：从初始学习率到目标学习率的增量
        lr_inc = (float(self.base_lr) - float(self.warmup_init_lr)) / float(self.warmup_steps)
        # 计算当前学习率
        lr = float(self.warmup_init_lr) + lr_inc * current_step
        return lr


# 余弦退火学习率调度器类，支持初期的线性预热
class CosineAnnealingLR:
    def __init__(self, lr, t_max, steps_per_epoch, max_epoch, warmup_epochs=0, eta_min=0):
        """
        初始化余弦退火调度器。

        参数:
        - lr: 基础学习率（最大学习率）。
        - t_max: 余弦退火的周期（epoch 数）。
        - steps_per_epoch: 每个 epoch 中的步骤数。
        - max_epoch: 总的 epoch 数。
        - warmup_epochs: 预热阶段的 epoch 数（默认为 0）。
        - eta_min: 最小学习率，退火阶段收敛至该值（默认为 0）。
        """
        self.base_lr = lr  # 最大学习率
        self.steps_per_epoch = steps_per_epoch  # 每个 epoch 的步数
        self.total_steps = int(max_epoch * steps_per_epoch)  # 总的训练步数
        self.T_max = t_max  # 余弦退火周期
        self.eta_min = eta_min  # 最小学习率
        # 初始化线性预热调度器对象
        self.warmup = _LinearWarmUp(lr, warmup_epochs, steps_per_epoch)

    def get_lr(self):
        """
        计算整个训练过程中的学习率变化，并返回每一步的学习率数组。

        返回:
        - 每步的学习率，作为一个 numpy 数组。
        """
        # 获取预热阶段的总步骤数
        warmup_steps = self.warmup.get_warmup_steps()

        lr_each_step = []  # 存储每一步的学习率
        current_lr = self.base_lr  # 初始学习率设为 base_lr

        # 计算每一步的学习率
        for i in range(self.total_steps):
            if i < warmup_steps:
                # 如果在预热阶段，使用线性增长的学习率
                lr = self.warmup.get_lr(i + 1)
            else:
                # 非预热阶段，应用余弦退火计算
                # 当前的 epoch 数
                cur_ep = i // self.steps_per_epoch
                # 每个 epoch 开始时重新计算当前的学习率
                if i % self.steps_per_epoch == 0 and i > 0:
                    current_lr = self.eta_min + \
                                 (self.base_lr - self.eta_min) * (1. + math.cos(math.pi * cur_ep / self.T_max)) / 2
                lr = current_lr  # 当前步的学习率为 current_lr

            # 记录每一步的学习率
            lr_each_step.append(lr)

        # 返回 numpy 数组形式的学习率序列，数据类型为 float32
        return np.array(lr_each_step).astype(np.float32)
