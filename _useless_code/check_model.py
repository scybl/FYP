from torchsummary import summary
import os



from LoadData.utils import load_config
from model_defination.model_loader import load_model_test

CONFIG_NAME = "config_train.yaml"
CONFIG_PATH = os.path.join("../configs/", CONFIG_NAME)
config = load_config(CONFIG_PATH)

model = load_model_test(config)
model = model.cuda()

summary(model, input_size=(3, 256, 256),device="cuda")  # 输入图像大小为 (3, 256, 256)
