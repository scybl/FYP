class PolyWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, initial_lr, power, eta_min):
        """
        Poly + Warmup 学习率调度器
        :param optimizer: 训练使用的优化器
        :param warmup_epochs: 多少轮次使用 warmup
        :param total_epochs: 总共的训练轮次
        :param initial_lr: 初始学习率
        :param power: poly decay 指数（通常 0.9）
        :param eta_min: 最小学习率
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.power = power
        self.eta_min = eta_min
        self.current_epoch = 0
        self.poly_factor = (1 - (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)) ** self.power

    def step(self):
        """
        在每个 epoch 结束后调用，调整学习率
        """
        if self.current_epoch < self.warmup_epochs:
            # **Warmup阶段：线性增加学习率**
            lr = self.eta_min + (self.initial_lr - self.eta_min) * (self.current_epoch / self.warmup_epochs)
        else:
            # **Poly Decay阶段**
            lr = (self.initial_lr - self.eta_min) * self.poly_factor + self.eta_min

        # **更新优化器中的学习率**
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1  # 更新当前 epoch 计数

    def get_lr(self):
        """返回当前的学习率"""
        return self.optimizer.param_groups[0]['lr']
