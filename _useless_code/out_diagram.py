import pandas as pd
import matplotlib.pyplot as plt

# TODO: this function have some bug, maybe fix later
# 读取CSV文件
loss_log_path = "../params/train_loss_log.csv"  # 替换为你的CSV文件路径
data = pd.read_csv(loss_log_path)

# 设置步长，例如每 100 个 step 显示一个
step_interval = 1000

# 绘制损失的折线图
plt.figure(figsize=(12, 6))
plt.plot(data['step'], data['train_loss'], label='Training Loss')
plt.xlabel("Step")
plt.ylabel("Training Loss")
plt.title("Training Loss per Step")
plt.legend()
plt.grid(True)

# 设置横坐标的步长
plt.xticks(range(0, len(data['step']), step_interval))

plt.show()
