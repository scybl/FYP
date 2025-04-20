import os
import torch
import numpy as np
import random
from torchvision.utils import save_image

from Evaluate.evaluate import dice, miou, binary_accuracy, binary_recall,binary_precision,binary_jaccard_index
from LoadData.data import get_dataset
from LoadData.utils import load_config
from Evaluate.LossChoose import LossFunctionHub
from model_defination.model_loader import load_model


# import warnings
# warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")

class Tester:

    def __init__(self, config_path, _model_name, _dataset_name):
        # 加载配置
        self.config = load_config(config_path)
        # 设置设备
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        # 获取类别数
        self.class_num = self.config["datasets"][_dataset_name]["class_num"]
        self.model_name = _model_name
        self.dataset_name = _dataset_name

        # 加载测试数据集，这里假设get_dataset支持"test"模式
        self.test_dataset = get_dataset(self.config, self.dataset_name, 'test')

        # 加载模型并移动到设备，注意测试时可能需要特殊设置，例如 batch_size 调整等
        self.net = load_model(self.config, _model_name, _dataset_name).to(self.device)

        self.save_image_path = self.config['save_image_path']

    def test(self) -> float:
        self.net.eval() # 设置为评估模式
        num_batches = 0

        dice_scores = []
        miou_scores = []
        accuracy_scores = []
        recall_scores = []
        precision_scores = []


        with torch.no_grad():
            for i, (image, segment_image) in enumerate(self.test_dataset):
                image, segment_image = image.to(self.device), segment_image.to(self.device)
                out_image = self.net(image)

                dice_scores.append(dice(pred=out_image,target=segment_image))
                miou_scores.append(miou(pred=out_image,target=segment_image))
                accuracy_scores.append(binary_accuracy(pred=out_image,target=segment_image))
                recall_scores.append(binary_recall(pred=out_image,target=segment_image))
                precision_scores.append(binary_precision(pred=out_image,target=segment_image))
                # jaccard_scores.append(binary_jaccard_index(pred=out_image,target=segment_image))

                num_batches += 1

                ####################################################################################
                # # 保存测试结果图像用于可视化
                # _image = image[0]
                # _segment_image = segment_image[0]
                # _out_image = out_image[0]

                # # 如果标签和输出为单通道，复制3通道方便可视化
                # _segment_image = _segment_image.repeat(3, 1, 1)
                # _out_image = _out_image.repeat(3, 1, 1)

                # # 堆叠原图、标签图、预测图
                # img = torch.stack([_image, _segment_image, _out_image], dim=0)
                # save_path = os.path.join(self.save_image_path,
                # f"{self.model_name}_{self.dataset_name}_{i}.png")
                # save_image(img, save_path)
                ####################################################################################
    
        _dice = sum(dice_scores) / num_batches
        _miou = sum(miou_scores) / num_batches
        _accuracy = sum(accuracy_scores) / num_batches
        _precision = sum(precision_scores) / num_batches
        _recall = sum(recall_scores) / num_batches

        return _dice,_miou,_accuracy,_precision,_recall


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    model_hub = ["bnet34"]
    dataset_hub = ['isic2018', 'clinicdb', 'kvasir']

    test_config_path = 'configs/config.yaml'

    repeat_times = 1
    seeds = [random.randint(1, 10000) for _ in range(repeat_times)]
    print(seeds)
    for model_name in model_hub:
        for dataset_name in dataset_hub:

            print("------------------------------------------")
            print(model_name + ' || ' + dataset_name)
            tester = Tester(test_config_path, _model_name=model_name, _dataset_name=dataset_name)
            dice_avg, miou_avg, acc_avg, prec_avg, recall_avg = 0, 0, 0, 0, 0

            for run in range(repeat_times):
                seed = seeds[run]
                set_random_seed(seed)
                print(f"Running {model_name} on {dataset_name}, Seed: {seed}")
                a,b,c,d,e= tester.test()

                # 假设 Tester.test() 返回上述指标的字典或元组，可调整如下
                # dice, miou, accuracy, precision, recall = tester.test()
                dice_avg += a
                miou_avg += b
                acc_avg += c
                prec_avg += d
                recall_avg += e

            print(f"Average Results for {model_name} on {dataset_name}:")
            print(f"Dice Avg: {dice_avg / repeat_times:.6f}")
            print(f"Miou Avg: {miou_avg / repeat_times:.6f}")
            print(f"Accuracy Avg: {acc_avg / repeat_times:.6f}")
            print(f"Precision Avg: {prec_avg / repeat_times:.6f}")
            print(f"Recall Avg: {recall_avg / repeat_times:.6f}")