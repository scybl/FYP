class PolyWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, initial_lr, power, eta_min):
        """
            Poly + Warmup learning rate scheduler
            :param optimizer: Optimizer used for training
            :param warmup_epochs: Number of epochs to use warmup
            :param total_epochs: Total number of training epochs
            :param initial_lr: Initial learning rate
            :param power: Poly decay exponent (usually 0.9)
            :param eta_min: Minimum learning rate
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
        if self.current_epoch < self.warmup_epochs:
            lr = self.eta_min + (self.initial_lr - self.eta_min) * (self.current_epoch / self.warmup_epochs)
        else:
            lr = (self.initial_lr - self.eta_min) * self.poly_factor + self.eta_min

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
