import torch.nn as nn
from monai.losses import DiceCELoss, DiceLoss, FocalLoss, TverskyLoss


"""
loss_hub = LossFunctionHub(loss_name="dice_ce", to_onehot_y=True, softmax=True)
loss_fn = loss_hub.get_loss_function()

loss_hub = LossFunctionHub(loss_name="cross_entropy", weight=None)
loss_fn = loss_hub.get_loss_function()
"""
class LossFunctionHub:
    def __init__(self, loss_name="dice_ce", **kwargs):
        """
        根据 loss_name 动态返回损失函数
        :param loss_name: 选择的损失函数，如 "dice_ce", "cross_entropy", "focal", "tversky"
        :param kwargs: 其他参数，如 class_weight, lambda_dice, lambda_ce 等
        """
        self.loss_name = loss_name.lower()
        self.kwargs = kwargs

    def get_loss_function(self):
        """ 返回指定的损失函数实例 """

        if self.loss_name == "dice_ce":
            return DiceCELoss(
                include_background=self.kwargs.get("include_background", True), # 是否计算背景
                to_onehot_y=self.kwargs.get("to_onehot_y", False),              # 是否将标签转换为one-hot格式
                sigmoid=self.kwargs.get("sigmoid", False),                      # 多类别-softmax，二分类 sigmoid
                softmax=self.kwargs.get("softmax", False),
                reduction=self.kwargs.get("reduction", "mean"),                 # 计算多个batch的平均损失
                lambda_dice=self.kwargs.get("lambda_dice", 0.7),                # dice loss 权重
                lambda_ce=self.kwargs.get("lambda_ce", 0.3),                    # cross entropy 权重
            )

        elif self.loss_name == "dice":
            return DiceLoss(
                include_background=self.kwargs.get("include_background", True),
                to_onehot_y=self.kwargs.get("to_onehot_y", False),
                sigmoid=self.kwargs.get("sigmoid", False),
                softmax=self.kwargs.get("softmax", False),
                reduction=self.kwargs.get("reduction", "mean"),
            )

        elif self.loss_name == "cross_entropy":
            return nn.CrossEntropyLoss(
                weight=self.kwargs.get("weight", None),
                reduction=self.kwargs.get("reduction", "mean"),
            )

        elif self.loss_name == "focal":
            return FocalLoss(
                gamma=self.kwargs.get("gamma", 2.0),
                weight=self.kwargs.get("weight", None),
                reduction=self.kwargs.get("reduction", "mean"),
            )

        elif self.loss_name == "tversky":
            return TverskyLoss(
                alpha=self.kwargs.get("alpha", 0.5),
                beta=self.kwargs.get("beta", 0.5),
                include_background=self.kwargs.get("include_background", True),
                smooth_nr=self.kwargs.get("smooth_nr", 1e-5),
                smooth_dr=self.kwargs.get("smooth_dr", 1e-5),
                to_onehot_y=self.kwargs.get("to_onehot_y", False),
                sigmoid=self.kwargs.get("sigmoid", False),
                softmax=self.kwargs.get("softmax", False),
                reduction=self.kwargs.get("reduction", "mean"),
            )

        else:
            raise ValueError(f"Unsupported loss function: {self.loss_name}")