import torch
import torch.nn.functional as F


def dice(pred, target, threshold=0.5, multiclass=False):
    """
    计算 Dice 系数（支持二分类和多分类）

    参数:
        pred: 预测 Tensor，shape=[B, C, H, W] 或 [B, 1, H, W]
        target: 真实标签 Tensor，shape=[B, H, W] 或 [B, 1, H, W]
        threshold: 二分类时的概率阈值
        multiclass: 是否为多分类任务

    返回:
        dice: Dice 系数，float
    """
    smooth = 1e-6
    pred = pred.float()
    target = target.float()

    if multiclass:
        # 多分类: pred shape = [B, C, H, W], target = [B, H, W]
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
        # 二分类: pred shape = [B, 1, H, W] or [B, H, W]
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
    计算 mean IoU（支持二分类和多分类）

    参数:
        pred: 预测 Tensor，shape=[B, C, H, W]（多分类）或 [B, 1, H, W]（二分类）
        target: 真实标签 Tensor，shape=[B, H, W] 或 [B, 1, H, W]
        threshold: 二分类时的概率阈值
        multiclass: 是否为多分类任务

    返回:
        mean_iou: float, mean IoU
    """
    eps = 1e-6
    pred = pred.float()
    target = target.float()
    
    if multiclass:
        # 多分类: pred shape = [B, C, H, W], target = [B, H, W]
        # 先取概率最高的类别作为预测结果
        pred_label = torch.argmax(pred, dim=1)  # [B, H, W]
        num_classes = int(pred_label.max().item()) + 1

        iou_sum = 0.0
        valid_classes = 0  # 防止某些类在 batch 中完全缺失
        for cls_id in range(num_classes):
            pred_cls = (pred_label == cls_id).float()  # [B, H, W]
            target_cls = (target == cls_id).float()    # [B, H, W]
            intersection = (pred_cls * target_cls).sum(dim=(1, 2))
            union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2)) - intersection
            # 只统计存在该类别的情况（防止出现 “某类在 batch 中没有标记” 引起不必要的干扰）
            # 如果 union 全是 0，说明 pred 和 target 都没有这个类，就不计入 IoU
            # 如果 target 有而 pred 没有，则 union != 0，需要计算
            iou_per_batch = (intersection + eps) / (union + eps)  # [B]
            
            # 只统计在 batch 中出现过该类(即 union>0)的样本
            valid_mask = (union > 0).float()
            # 防止全 batch 都没有该类，导致分母是 0
            if valid_mask.sum() > 0:
                iou_sum += (iou_per_batch * valid_mask).sum() / valid_mask.sum()
                valid_classes += 1
        
        if valid_classes == 0:
            return 1.0  # 如果所有类都没出现，直接返回 1.0
        else:
            return iou_sum / valid_classes
    
    else:
        # 二分类: pred shape = [B, 1, H, W] or [B, H, W], target shape 同理
        if pred.dim() == 4:
            pred = pred.squeeze(1)  # [B, H, W]
        if target.dim() == 4:
            target = target.squeeze(1)  # [B, H, W]

        # 根据 threshold 将预测转为二值
        pred_bin = (pred > threshold).float()
        
        intersection = (pred_bin * target).sum(dim=(1, 2))
        union = pred_bin.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
        iou = (intersection + eps) / (union + eps)  # [B]
        return iou.mean()


def binary_jaccard_index(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    计算二分类 Jaccard Index (IoU)
    
    参数:
        pred: 预测概率或 logit，形状 [B, 1, H, W] 或 [B, H, W]
        target: 真实标签，形状 [B, 1, H, W] 或 [B, H, W]，元素为 {0,1}
        threshold: 将预测概率转换为二值的阈值

    返回:
        scalar Tensor，表示二分类 Jaccard Index 的 batch 平均值
    """
    eps = 1e-6
    # 保证 pred, target 的维度一致
    if pred.dim() == 4:
        pred = pred.squeeze(1)  # 变为 [B, H, W]
    if target.dim() == 4:
        target = target.squeeze(1)  # 变为 [B, H, W]

    # 将预测二值化
    pred_bin = (pred > threshold).float()
    target_bin = target.float()

    # 计算交集与并集
    intersection = (pred_bin * target_bin).sum(dim=(1, 2))  # [B]
    union = pred_bin.sum(dim=(1, 2)) + target_bin.sum(dim=(1, 2)) - intersection  # [B]

    # Jaccard Index = intersection / union
    iou = (intersection + eps) / (union + eps)  # [B]
    return iou.mean()  # batch 平均


def binary_precision(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    计算二分类 Precision = TP / (TP + FP)

    参数:
        pred: 预测概率或 logit，形状 [B, 1, H, W] 或 [B, H, W]
        target: 真实标签，形状 [B, 1, H, W] 或 [B, H, W]，元素为 {0,1}
        threshold: 将预测概率转换为二值的阈值

    返回:
        scalar Tensor，表示二分类 Precision 的 batch 平均值
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
    # 预测为正的总数 (TP + FP)
    pred_positives = pred_bin.sum(dim=(1, 2))  # [B]

    precision = (tp + eps) / (pred_positives + eps)  # [B]
    return precision.mean()


def binary_recall(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    计算二分类 Recall = TP / (TP + FN)

    参数:
        pred: 预测概率或 logit，形状 [B, 1, H, W] 或 [B, H, W]
        target: 真实标签，形状 [B, 1, H, W] 或 [B, H, W]，元素为 {0,1}
        threshold: 将预测概率转换为二值的阈值

    返回:
        scalar Tensor，表示二分类 Recall 的 batch 平均值
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
    # 实际为正的总数 (TP + FN)
    actual_positives = target_bin.sum(dim=(1, 2))  # [B]

    recall = (tp + eps) / (actual_positives + eps)
    return recall.mean()


def binary_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    计算二分类的 Accuracy = (TP + TN) / (所有像素数)

    参数:
        pred: 预测概率或 logit，形状 [B, 1, H, W] 或 [B, H, W]
        target: 真实标签，形状 [B, 1, H, W] 或 [B, H, W]，元素为 {0,1}
        threshold: 将预测概率转换为二值的阈值

    返回:
        scalar Tensor，表示二分类准确率的 batch 平均值
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
    # 注意：correct[0].numel() = H*W, 对于 batch 里的每一张图像都一样

    return accuracy_per_image.mean()