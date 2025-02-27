# TODO：使用transunet的数据集加载来获取数据集
# TODO：学习transunet的enocder部分, 作一下encoder的部分，调用一个预训练模型

# Poly Strategy, H2Former，
# warmup, 训练 K 轮之后，lr再降低，
# Spark: https://github.com/keyu-tian/SparK/blob/main/pretrain/utils/lr_control.py, warm-up 锁定，K轮次之后再降低lr
# TODO: 深度监督的处理方法
# TODO: 添加一个划分验证集，测试集，训练集的方法，根据config的配置来划分
# TODO: 修复test class

# TODO: kvasir和clinicDB数据集训练出来结果全黑,需要完整训练测试


# TODO: 查看h5文件,在终端输入vitables,然后打开文件(已解决)
# TODO：更改学习率的部分，这里有问题,(已解决)