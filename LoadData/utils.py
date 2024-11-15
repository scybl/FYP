from PIL import Image
import yaml

"""这个方法是用来进行等比缩放的，取每个图像的最长边，做一个max矩形，将这个原图贴到这个矩形上就完成了一个等比缩放"""


def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)  # read the img file
    temp = max(img.size)  # get the max edge size
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


# 加载配置文件的函数
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
