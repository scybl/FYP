import os
import torch
from torchvision.utils import save_image

from Evaluate.evaluate import dice
from LoadData.data import get_dataset
from LoadData.utils import load_config
from Evaluate.LearningRate import PolyWarmupScheduler
from Evaluate.LossChoose import LossFunctionHub
from model.model_loader import load_model
from torch.optim import AdamW

from fvcore.nn import FlopCountAnalysis, parameter_count_table


class Trainer:
    def __init__(self, config_path, model_name, dataset_name):
        self.config = load_config(config_path)
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")
        self.class_num = self.config["datasets"][dataset_name]["class_num"]
        self.model_name = model_name

        self.dataset_name = dataset_name
        self.train_dataset = get_dataset(self.config, self.dataset_name, 'train')
        self.val_dataset = get_dataset(self.config, self.dataset_name, 'val')

        self.net = load_model(self.config, model_name, dataset_name).to(self.device)

        if self.class_num == 1:
            loss_hub = LossFunctionHub(loss_name="dice_ce", include_background=False, to_onehot_y=False, softmax=False,
                                       sigmoid=True)  # single
        else:
            loss_hub = LossFunctionHub(loss_name="dice_ce", include_background=True, to_onehot_y=False, softmax=True,
                                       sigmoid=False)  # multi-classes

        self.loss_fn = loss_hub.get_loss_function()

        self.opt = AdamW(self.net.parameters(), lr=self.config["setting"]['min_lr'],
                         betas=(0.99, 0.95))
        self.scheduler = PolyWarmupScheduler(
            optimizer=self.opt,
            warmup_epochs=self.config["setting"]['warmup_epochs'],
            total_epochs=self.config["setting"]['epochs'],
            initial_lr=self.config["setting"]['max_lr'],
            power=0.9,
            eta_min=self.config["setting"]['min_lr']
        )
        self.save_model_path = os.path.join(self.config['model']["save_path"], model_name)
        self.loss_log_path = os.path.join(self.config['model']['save_path'],
                                          f"log_{model_name}_{self.dataset_name}.csv")
        self._init_log_file()

    def _init_log_file(self):
        file_exists = os.path.exists(self.loss_log_path)
        mode = "a" if file_exists else "w"
        with open(self.loss_log_path, mode) as f:
            if not file_exists:
                f.write("epoch,step,train_loss\n")

    def val(self) -> float:
        self.net.eval()  # evaluate
        num_batches = 0
        dice_scores = []

        with torch.no_grad():
            for i, (image, segment_image) in enumerate(self.val_dataset):
                image, segment_image = image.to(self.device), segment_image.to(self.device)
                out_image = torch.sigmoid(self.net(image))  # sigmoid

                # Dice
                dice_score = dice(pred=out_image, target=segment_image)
                dice_scores.append(dice_score)

                num_batches += 1

        # 计算平均 Dice
        avg_dice = sum(dice_scores) / num_batches
        print(f"Validation Dice: {avg_dice:.6f}")
        return avg_dice

    def train(self):
        epochs = 1
        best_val = self.val()

        while epochs <= self.config["setting"]['epochs']:
            self.net.train()  # 确保模型处于训练模式
            for i, (image, segment_image) in enumerate(self.train_dataset):
                self.opt.zero_grad()
                image, segment_image = image.to(self.device), segment_image.to(self.device)
                out_image = self.net(image)
                # print(f'The size of the image is: f{image.size()}')
                # print(f'The size of the mask is: f{segment_image.size()}')
                # print(f'The size of the out_img is: f{out_image.size()}')

                train_loss = self.loss_fn(out_image, segment_image)

                train_loss.backward()
                self.opt.step()

                # save the logging
                with open(self.loss_log_path, "a") as f:
                    f.write(f"{epochs},{i},{train_loss.item():.6f}\n")

                current_lr = self.opt.param_groups[0]['lr']
                print(f"Epoch {epochs} --- Step {i} --- Loss: {train_loss.item():.6f} --- LR: {current_lr:.6f}")

            val_dice = self.val()

            # save the best
            if val_dice > best_val:
                best_val = val_dice
                torch.save(self.net.state_dict(), f"{self.save_model_path}_{self.dataset_name}_best.pth")
                print(f"Epoch {epochs}: save the best model")

            # update the lr
            self.scheduler.step()
            epochs += 1

    def analyze(self, input_tensor_size):
        """
            Calculate the FLOPs and number of parameters of the model based on
            the input tensor size and print the results.

            Parameters:
            input_tensor_size (tuple): Input tensor size, format (Channels, Height, Width)
        """
        # Construct a random tensor that matches the model input (Batch size defaults to 1)

        input_tensor = torch.randn(1, *input_tensor_size).to(self.device)

        flops = FlopCountAnalysis(self.net, input_tensor)
        print("Total FLOPs:", flops.total())

        print("\nParameters:")
        print(parameter_count_table(self.net))


# run
if __name__ == "__main__":

    model_hub = [
        "duck",
        "unetpp",
        "bnet",
        'unet',
        "bnet34",
        'unext',
        'dga',
        'pham'
    ]
    dataset_hub = [
        'kvasir',
        'clinicdb',
        'isic2018',
        # 'synapse'
    ]

    train_config_path = 'configs/config.yaml'
    for model_name in model_hub:
        for dataset_name in dataset_hub:
            print('-----------------')
            trainer = Trainer(train_config_path, model_name=model_name, dataset_name=dataset_name)
            trainer.analyze((3, 224, 224))
            # trainer.train()
