# FlameWork

以下是我的项目目录，以及相关的comments
```markdown
FYP/
    └── _useless_code											Save the useless code, could ignore
    └── Block													NN block, if unnecessary, no change
    └── configs
        └── config.yaml											KEY: control the whole project params
    └── LoadData
        └── data.py												Load the data, controlled by the config.yaml
        └── utils.py											Improve the support function for data.py
    └── model_defination										All the model
        └── MyFrame												**The project model designned by myself**
        └── LoadModel.py										Load model function, controlled by configs.yaml
        └── UnetBasethe												
        └── Unext   													
        └── Unetpp  													
        └── SeNet   													
        └── DenseNetDensely 									
        └── ResNet														
        └── fcn_s8  													
        └── vgg16   													
    └── test_and_train											The code about test and train
        └── cosineannealingLR.py							    Cosine annealing function, return the lr
        └── test_model.py										Test the model
        └── out_diagram.py										
        └── train_model.py										
    └── FLOPs.py												Calculate the params count and calculate count
    └── README.md															

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
## 11.18
我需要考虑一种能同时关注空间特征和通道特征的模块，这个模块可以根据输入的特征图，动态的平衡空间特征和通道特征的方法
## 11.20

problem 在训练的过程中，模型经常过拟合，在epoch30左右的时候就会过拟合不知道为什么

solution:：所有设计的模型都必须考虑所用数据集的size，也要考虑全尺寸模型动态等方面

## 11.21

Proble: 我发现一般的模型设计都是针对于图像识别的模块，如果我直接调用预训练模型可能并没有办法直接使用，例如resnet他引入的残差块，他在设计的时候是用来做图像识别的，但是我做图像分割的话后面可能会再做一个decoder的部分，但是预训练模型是没有这个部分，我该怎么做

## 11.26

需要实现一下nnunet模型还有unet+++

# 图像分割损失函数 & 评估指标总结

| **名称** | **类型** | **适用角度** | **主要作用** | **适用场景** | **优缺点** |
|------------|----------|--------------|----------------------|--------------|----------------------|
| **Dice 相似系数 (DSC)** | 评估指标 | 局部效果 | 衡量预测与真实分割的重叠度 | 适用于总体评估 | 适用于类别不平衡，但对细节不敏感 |
| **IoU (Jaccard Index)** | 评估指标 | 局部效果 | 计算预测和真实区域的交并比 | 适用于总体评估 | 比 DSC 更严格，低 IoU 说明误差较大 |
| **Pixel Accuracy** | 评估指标 | 局部效果 | 计算正确分类像素的比例 | 适用于均衡数据 | 对类别不平衡不敏感，容易导致误导性结果 |
| **Hausdorff 距离 (HD)** | 评估指标 | 边缘效果 | 计算预测分割与真实分割的最大误差 | 适用于边界匹配 | 受异常值影响较大 |
| **HD95** | 评估指标 | 边缘效果 & 极端情况 | 计算 95% 分位数的 Hausdorff 距离 | 适用于边界评估，避免异常值干扰 | 比 HD 更稳定，更能反映整体误差情况 |
| **ASSD (平均对称表面距离)** | 评估指标 | 边缘效果 | 计算预测边界和真实边界的平均对称距离 | 适用于边界评估 | 适合评价局部边界误差，但不能完全代表全局误差 |
| **Boundary Loss** | 损失函数 | 边缘效果 | 专注于优化边界像素 | 适用于医学图像和细粒度分割 | 需要结合其他损失使用，以保证整体分割质量 |
| **DiceCE (Dice + Cross-Entropy)** | 损失函数 | 局部效果 & 边缘效果 | 结合 Dice Loss 和 Cross-Entropy Loss，兼顾全局和局部 | 适用于医学图像 & 小目标分割 | 解决类别不平衡问题，同时优化全局和局部，但计算量略大 |
| **Tversky Loss** | 损失函数 | 极端情况 | 通过调节 α 和 β 控制 FP & FN 的影响 | 适用于不平衡类别 | 可调节权重，但需手动设置超参数 |
| **Focal Loss** | 损失函数 | 极端情况 | 减少易分类样本的影响，专注于难分类样本 | 适用于小目标 & 不平衡数据 | 适合目标检测，可能影响全局稳定性 |
| **Lovász-Softmax Loss** | 损失函数 | 极端情况 | 直接优化 IoU，提升低 IoU 样本的优化效果 | 适用于小目标、破碎分割 | 优化 IoU 但计算复杂度较高 |