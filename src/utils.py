from torchmetrics.functional import dice
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

def calculate_metrics_v1(preds_argmax, targets, n_classes, device):
    iou_metric = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    iou = iou_metric(preds_argmax, targets)

    dice_score = dice(preds_argmax, targets, num_classes=n_classes, average='micro')

    return iou, dice_score


def calculate_metrics_v2(preds_argmax, targets, n_classes, device):
    acc_metric = MulticlassAccuracy(num_classes=n_classes, average="none").to(device)
    acc = acc_metric(preds_argmax, targets)

    return acc