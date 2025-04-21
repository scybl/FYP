import torch
import torch.nn.functional as F


def dice(pred, target, threshold=0.5, multiclass=False):
    """
        Calculate the Dice coefficient (supports binary and multi-class classification)

        Parameters:
            pred: Predicted Tensor, shape=[B, C, H, W] or [B, 1, H, W]
            target: Ground truth Tensor, shape=[B, H, W] or [B, 1, H, W]
            threshold: Probability threshold for binary classification
            multiclass: Whether it's a multi-class task

        Returns:
            dice: Dice coefficient, float
    """
    smooth = 1e-6
    pred = pred.float()
    target = target.float()

    if multiclass:
        # Multiclass classification
        pred = torch.argmax(pred, dim=1)  # [B, H, W]
        num_classes = pred.max().item() + 1
        dice_total = 0.0
        for cls in range(num_classes):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()
            intersection = (pred_cls * target_cls).sum(dim=(1, 2))
            union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
            dice = (2. * intersection + smooth) / (union + smooth)
            dice_total += dice.mean()
        return dice_total / num_classes
    else:
        # binary: pred shape = [B, 1, H, W] or [B, H, W]
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        pred_bin = (pred > threshold).float()

        intersection = (pred_bin * target).sum(dim=(1, 2))
        union = pred_bin.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.mean()


def miou(pred, target, threshold=0.5, multiclass=False):
    """
        Calculate mean IoU (supports binary and multi-class classification)

        Parameters:
            pred: Predicted Tensor, shape=[B, C, H, W] (multi-class) or [B, 1, H, W] (binary classification)
            target: Ground truth Tensor, shape=[B, H, W] or [B, 1, H, W]
            threshold: Probability threshold for binary classification
            multiclass: Whether it's a multi-class task

        Returns:
            mean_iou: float, mean IoU
    """
    eps = 1e-6
    pred = pred.float()
    target = target.float()

    if multiclass:
        pred_label = torch.argmax(pred, dim=1)  # [B, H, W]
        num_classes = int(pred_label.max().item()) + 1

        iou_sum = 0.0
        valid_classes = 0
        for cls_id in range(num_classes):
            pred_cls = (pred_label == cls_id).float()  # [B, H, W]
            target_cls = (target == cls_id).float()  # [B, H, W]
            intersection = (pred_cls * target_cls).sum(dim=(1, 2))
            union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2)) - intersection
            iou_per_batch = (intersection + eps) / (union + eps)  # [B]

            valid_mask = (union > 0).float()
            if valid_mask.sum() > 0:
                iou_sum += (iou_per_batch * valid_mask).sum() / valid_mask.sum()
                valid_classes += 1

        if valid_classes == 0:
            return 1.0
        else:
            return iou_sum / valid_classes

    else:
        if pred.dim() == 4:
            pred = pred.squeeze(1)  # [B, H, W]
        if target.dim() == 4:
            target = target.squeeze(1)  # [B, H, W]

        pred_bin = (pred > threshold).float()

        intersection = (pred_bin * target).sum(dim=(1, 2))
        union = pred_bin.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
        iou = (intersection + eps) / (union + eps)  # [B]
        return iou.mean()


def binary_jaccard_index(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
        Calculate the binary classification Jaccard Index (IoU)

        Parameters:
            pred: Predicted probabilities or logits, shape=[B, 1, H, W] or [B, H, W]
            target: Ground truth labels, shape=[B, 1, H, W] or [B, H, W], with elements {0, 1}
            threshold: Threshold to convert predicted probabilities to binary values

        Returns:
            scalar Tensor, representing the batch average of the binary classification Jaccard Index
    """

    eps = 1e-6
    if pred.dim() == 4:
        pred = pred.squeeze(1)  # [B, H, W]
    if target.dim() == 4:
        target = target.squeeze(1)  # [B, H, W]

    pred_bin = (pred > threshold).float()
    target_bin = target.float()

    intersection = (pred_bin * target_bin).sum(dim=(1, 2))  # [B]
    union = pred_bin.sum(dim=(1, 2)) + target_bin.sum(dim=(1, 2)) - intersection  # [B]

    # Jaccard Index = intersection / union
    iou = (intersection + eps) / (union + eps)  # [B]
    return iou.mean()  # batch mean


def binary_precision(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
        Calculate binary classification Precision = TP / (TP + FP)

        Parameters:
            pred: Predicted probabilities or logits, shape=[B, 1, H, W] or [B, H, W]
            target: Ground truth labels, shape=[B, 1, H, W] or [B, H, W], with elements {0, 1}
            threshold: Threshold to convert predicted probabilities to binary values

        Returns:
            scalar Tensor, representing the batch average of the binary classification Precision
    """
    eps = 1e-6
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    pred_bin = (pred > threshold).float()
    target_bin = target.float()

    # TP = pred=1 & target=1
    tp = (pred_bin * target_bin).sum(dim=(1, 2))  # [B]
    # predict (TP + FP)
    pred_positives = pred_bin.sum(dim=(1, 2))  # [B]

    precision = (tp + eps) / (pred_positives + eps)  # [B]
    return precision.mean()


def binary_recall(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
        Calculate binary classification Recall = TP / (TP + FN)

        Parameters:
            pred: Predicted probabilities or logits, shape=[B, 1, H, W] or [B, H, W]
            target: Ground truth labels, shape=[B, 1, H, W] or [B, H, W], with elements {0, 1}
            threshold: Threshold to convert predicted probabilities to binary values

        Returns:
            scalar Tensor, representing the batch average of the binary classification Recall
    """
    eps = 1e-6
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    pred_bin = (pred > threshold).float()
    target_bin = target.float()

    # TP = pred=1 & target=1
    tp = (pred_bin * target_bin).sum(dim=(1, 2))  # [B]
    # TP + FN
    actual_positives = target_bin.sum(dim=(1, 2))  # [B]

    recall = (tp + eps) / (actual_positives + eps)
    return recall.mean()


def binary_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
        Calculate binary classification Accuracy = (TP + TN) / (Total number of pixels)

        Parameters:
            pred: Predicted probabilities or logits, shape=[B, 1, H, W] or [B, H, W]
            target: Ground truth labels, shape=[B, 1, H, W] or [B, H, W], with elements {0, 1}
            threshold: Threshold to convert predicted probabilities to binary values

        Returns:
            scalar Tensor, representing the batch average of the binary classification Accuracy
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    pred_bin = (pred > threshold).float()
    target_bin = target.float()

    # TP = (pred=1 & target=1)
    tp = (pred_bin * target_bin)
    # TN = (pred=0 & target=0)
    tn = ((1 - pred_bin) * (1 - target_bin))

    correct = tp + tn  # [B, H, W]
    accuracy_per_image = correct.sum(dim=(1, 2)) / correct[0].numel()  # [B]

    return accuracy_per_image.mean()
