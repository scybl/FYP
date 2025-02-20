from PIL import Image
import yaml

# 加载配置文件的函数
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
