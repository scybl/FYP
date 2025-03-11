import numpy as np
import matplotlib.pyplot as plt

"""
这个方法同来绘制学习率图，用于报告
"""


# 1. 线性 warmup 方法
def linear_warmup(steps, max_lr=1.0, min_lr=0.0):
    # 初始化损失值数组
    loss = np.zeros_like(steps, dtype=float)

    for i, step in enumerate(steps):
        if step <= 20:
            # 第一条直线，学习率从 0 增加到 max_lr
            loss[i] = max_lr * (step / 20)
        else:
            # 第二条直线，学习率从 max_lr 下降到 min_lr
            loss[i] = max_lr - (step - 20) * (max_lr - min_lr) / 80

    return loss


# 2. 余弦 warmup
def cosine_warmup(steps, max_lr=1.0, min_lr=0.0):
    # 初始化损失值数组
    loss = np.zeros_like(steps, dtype=float)

    for i, step in enumerate(steps):
        if step <= 20:
            # 第一阶段：从 (0, 0) 到 (20, 1.0) 的线性变化
            loss[i] = step / 20
        else:
            # 第二阶段：余弦衰减函数，周期为80
            loss[i] = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * (step - 20) / 80))

    return loss


# 3. 周期余弦 warmup
def cyclical_cosine_warmup(steps, max_lr=1.0, min_lr=0.0, cycle_length=40):
    # 初始化损失值数组
    loss = np.zeros_like(steps, dtype=float)

    for i, step in enumerate(steps):
        if step <= 20:
            # 第一阶段：从 (0, 0) 到 (20, 1.0) 的线性变化
            loss[i] = step / 20
        else:
            # 计算当前周期的相对步数
            cycle_position = ((step - 20) % cycle_length) / cycle_length
            # 使用余弦函数来计算学习率
            loss[i] = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * cycle_position))

    return loss


# 4. 指数 warmup
def exponential_warmup(steps, max_lr=1.0, decay_rate=0.1):
    # 初始化损失值数组
    loss = np.zeros_like(steps, dtype=float)

    for i, step in enumerate(steps):
        if step <= 20:
            # 第一阶段：从 (0, 0) 到 (20, 1.0) 的线性变化
            loss[i] = step / 20
        else:
            # 第二阶段：指数级衰减，从 (20, 1.0) 到 (100, 0.0)
            decay_factor = np.exp(-decay_rate * (step - 20))
            loss[i] = max_lr * decay_factor

    return loss


# 5. 阶梯 warmup
# 定义阶梯衰减的 warmup 方法
def step_warmup(steps, max_lr=1.0, step_size=20):
    # 初始化损失值数组
    loss = np.zeros_like(steps, dtype=float)

    for i, step in enumerate(steps):
        if step <= 20:
            # 第一阶段：从 (0, 0) 到 (20, 1.0) 的线性变化
            loss[i] = step / 20
        else:
            # 第二阶段：阶梯衰减，每 step_size 个步骤衰减一次，每次减半
            decay_factor = 0.5 ** (step // step_size)
            loss[i] = max_lr * decay_factor
    return loss


def poly_warmup(steps, max_lr=1.0, warmup_steps=20, total_steps=100, poly_power=2):
    """
    前 warmup_steps 步：线性预热，从 0 增加到 max_lr
    后 (total_steps - warmup_steps) 步：采用多项式衰减策略，公式：
        lr = max_lr * (1 - (step - warmup_steps) / (total_steps - warmup_steps)) ** poly_power
    """
    loss = np.zeros_like(steps, dtype=float)

    for i, step in enumerate(steps):
        if step <= warmup_steps:
            loss[i] = (step / warmup_steps) * max_lr
        else:
            effective_step = step - warmup_steps
            decay_steps = total_steps - warmup_steps
            loss[i] = max_lr * (1 - effective_step / decay_steps) ** poly_power
    return loss


# 设置步数和学习率范围
steps = np.arange(0, 100)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制不同的学习率调整策略
plt.plot(steps, linear_warmup(steps), label='Linear', color='orange')
plt.plot(steps, cosine_warmup(steps), label='Cosine', color='blue')
plt.plot(steps, cyclical_cosine_warmup(steps), label='Cyclical Cos', color='green')
plt.plot(steps, exponential_warmup(steps), label='Exponential', color='red')
plt.plot(steps, step_warmup(steps), label='Steps', color='purple')
plt.plot(steps, poly_warmup(steps), label='Poly', color='black')

# 添加标签
plt.xlabel('Steps')
plt.ylabel('Learning Rate')
plt.title('Warmup with Different Decay Strategies')
plt.legend()

# 显示图形
plt.show()
