# FlameWork

currently, I have implement the following model
```
.dataset/
    └──benchmark_RELEASE
        └──dataset
            └──train.txt                  8498行图像的名字索引
            └──cls                        11355全是mat文件
            └──img                        11355全是jpg图像文件（隶属于voc的JPEGImage文件夹下的17125张jpg图像）
            └──inst                       11355全是mat文件
            └──val.txt                    2857行图像的名字索引
    └──VOCdevkit
        └── VOC2012
            ├── Annotations               所有的图像标注信息(XML文件)
            ├── ImageSets    
            │   ├── Action                人的行为动作图像信息
            │   ├── Layout                人的各个部位图像信息
            │   │
            │   ├── Main                  目标检测分类图像信息
            │   │     ├── train.txt       训练集(5717)
            │   │     ├── val.txt         验证集(5823)
            │   │     └── trainval.txt    训练集+验证集(11540)
            │   │
            │   └── Segmentation          目标分割图像信息
            │         ├── train.txt       训练集(1464)
            │         ├── val.txt         验证集(1449)
            │         └── trainval.txt    训练集+验证集(2913)
            │ 
            ├── JPEGImages                所有图像文件
            ├── SegmentationClass         语义分割png图（基于类别）
            └── SegmentationObject        实例分割png图（基于目标）


```

this is my own python code architecture to control all the model
according to change the test model

1. base on the seed to control the control data load