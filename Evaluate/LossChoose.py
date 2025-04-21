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
            Dynamically return the loss function based on loss_name
            :param loss_name: Selected loss function, such as "dice_ce", "cross_entropy", "focal", "tversky"
            :param kwargs: Other parameters, such as class_weight, lambda_dice, lambda_ce, etc.
        """
        self.loss_name = loss_name.lower()
        self.kwargs = kwargs

    def get_loss_function(self):

        if self.loss_name == "dice_ce":
            return DiceCELoss(
                include_background=self.kwargs.get("include_background", True),
                to_onehot_y=self.kwargs.get("to_onehot_y", False),
                sigmoid=self.kwargs.get("sigmoid", False),
                softmax=self.kwargs.get("softmax", False),
                reduction=self.kwargs.get("reduction", "mean"),
                lambda_dice=self.kwargs.get("lambda_dice", 0.5),
                lambda_ce=self.kwargs.get("lambda_ce", 0.5),
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

        elif self.loss_name == "bce":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_name}")
