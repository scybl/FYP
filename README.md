# FlameWork

以下是我的项目目录，以及相关的comments
```markdown
FYP/
    └── _useless_code													Save the useless code, could ignore
    └── Block																	NN block, if unnecessary, no change
    └── configs
        └── config.yaml												KEY: control the whole project params
    └── LoadData
        └── data.py														Load the data, controlled by the config.yaml
        └── utils.py													Improve the support function for data.py
    └── model_defination											All the model
        └── MyFrame														**The project model designned by myself**
        └── LoadModel.py											Load model function, controlled by configs.yaml
        └── UnetBasethe												Functional unet model
        └── Unext   													Advance Unet -> Unext
        └── Unetpp  													Unet Plus Plus
        └── SeNet   													Squeeze-and-Excitation Networks
        └── DenseNetDensely 									Connected Convolutional Networks
        └── ResNet														Residual Neural networks(50, 101, 152)
        └── fcn_s8  													Fully Convolutional Networks
        └── vgg16   													Fully Convolutional Networks
    └── test_and_train												The code about test and train
        └── cosineannealingLR.py							Cosine annealing function
        └── test_model.py											Test the model
        └── out_diagram.py										Base on the test and train model, Output image
        └── train_model.py										Train the model, controlled by config.yaml
    └── FLOPs.py															Calculate the params count and calculate count
    └── README.md															Read me file

```

在你执行我的项目之前，您应该先执行以下的脚本，用来配置环境以辅助项目运行

```shell
#!/bin/bash

# Define the directory names
DIRS=("saved_images" "params")

# Loop through the directory names
for DIR_NAME in "${DIRS[@]}"; do
  # Check if the directory exists
  if [ ! -d "$DIR_NAME" ]; then
    # Create the directory
    mkdir "$DIR_NAME"
    echo "Directory '$DIR_NAME' has been created."
  else
    echo "Directory '$DIR_NAME' already exists."
  fi
done
```

# Record

this section will record the problem i have, and the solution about it

## 10.20

problem：

​	在训练过程中，我发现显卡并不能跑满，不知道问题在哪

solution:

​	在训练过程中没有设置nums of works， 默认为1 肯定跑不快

## 11.9

problem： 在训练transunet的时候，不知道为什么我的训练结果总是低于paper中所的结果

solution： 我使用的数据集是isic2018， 这个数据集太小了，在使用的时候直接过拟合了，所以测试结果不佳

## 11.16

Problem: 

​	在读论文的过程中，不知道为什么这个模型训练的时候是一个阶梯下降的

<img src="/Users/libingze/Desktop/Bizzarr_Code/FYP/imgs/resLr.png" alt="resLr" style="zoom:25%;" />

solution: 

​	是在训练的时候才用了动态学习率的方法, 所以在后面我训练的时候也可以考虑这个，我准备使用**余弦退火**
$$
v_t = \beta v_{t-1} + (1 - \beta) \left( \nabla L(w_t) + \lambda w_t \right)  \\

w_{t+1} = w_t - \eta v_t \\

 v_t ：当前的动量，用于加速收敛并减小振荡。\\

 \eta：动态调整的学习率。\\

\lambda ：权重衰减系数，用于正则化。
$$

## 11.20

problem 在训练的过程中，模型经常过拟合，在epoch30左右的时候就会过拟合不知道为什么

solution:：所有设计的模型都必须考虑所用数据集的size，也要考虑全尺寸模型动态等方面

## 11.21

Proble: 我发现一般的模型设计都是针对于图像识别的模块，如果我直接调用预训练模型可能并没有办法直接使用，例如resnet他引入的残差块，他在设计的时候是用来做图像识别的，但是我做图像分割的话后面可能会再做一个decoder的部分，但是预训练模型是没有这个部分，我该怎么做

## 11.26

需要实现一下nnunet模型还有unet+++