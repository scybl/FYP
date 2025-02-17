import time
import torch
from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    def __init__(self, size=10000):
        self.data = torch.randn(size, 3, 224, 224)  # 模拟图像数据
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        time.sleep(0.01)  # 模拟数据加载延迟
        return self.data[idx]

# 添加更多的 num_workers 测试点
num_workers_list = [0, 2, 4, 8, 12, 16, 24, 32, 36, 40, 44]
results = {}

for num_workers in num_workers_list:
    dataloader = DataLoader(DummyDataset(), batch_size=32, num_workers=num_workers, pin_memory=True)
    
    start_time = time.time()
    for batch in dataloader:
        pass  # 模拟训练
    elapsed_time = time.time() - start_time
    results[num_workers] = elapsed_time
    print(f"num_workers={num_workers}, Time={elapsed_time:.2f} sec")

# 找到最优 num_workers
best_workers = min(results, key=results.get)
print(f"Best num_workers: {best_workers}, Time={results[best_workers]:.2f} sec")