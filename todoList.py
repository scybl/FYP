# Question: 针对于肠息肉数据,色彩信息能够提供一定的分割信息，
# 但是相比之下，肠息肉的形状、纹理信息对于分割而言则是更加重要的特征来源。

# TODO：使用transunet的数据集加载来获取数据集
# TODO：学习transunet的enocder部分, 作一下encoder的部分，调用一个预训练模型
# 添加res encoder并将原本封装为original class


# TODO: 深度监督的处理方法
# TODO: 添加一个划分验证集，测试集，训练集
# TODO: 修复test class
# TODO: 最近发现针对于分割任务,他的边缘问题应该着重注意,边缘平滑,边缘模糊问题对于模型影响很大
# TODO: kvasir和clinicDB数据集训练出来结果全黑,需要完整训练测试
"""
数据质量问题：数据集中可能存在噪音、图像质量差或者标签不准确等问题，这会导致模型学习到错误的特征或者无法正确学习到目标结构。
数据量不足：数据过少
数据不平衡：数据集中正例和负例之间的分布不平衡，模型可能会倾向于预测更常见的类别，而忽略较少出现的类别
数据预处理不当：图像尺寸的标准化、数据增强、图像配准等预处理步骤可以提高模型的鲁棒性和泛化能力。
超参数选择不当：学习率、优化器、损失函数等选择都会影响模型的性能
"""

# 进度
# TODO: 查看h5文件,在终端输入vitables,然后打开文件(已解决)
# TODO：更改学习率的部分，这里有问题,(已解决)
# 制作一个line table来说明这个
# TODO：更改动态学习率（已完成）
# Poly Strategy, H2Former，
# warmup, 训练 K 轮之后，lr再降低，
# Spark: https://github.com/keyu-tian/SparK/blob/main/pretrain/utils/lr_control.py, warm-up 锁定，K轮次之后再降低lr