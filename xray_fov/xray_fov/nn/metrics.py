"""author: Alexander Ortlieb"""

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from torch.nn import BCEWithLogitsLoss
import torch
from xray_fov.nn.boundingbox import metric_largest_box_iou

def tensor_to_float(tensor):
    return float(tensor.cpu().detach().numpy())

def check_accuracy(epoch, loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    hausdorff_distance = 0
    BCEL = 0
    n = len(loader.sampler)
    metric_hd = HausdorffDistanceMetric(percentile=95)
    metric_hd.reset()
    metric_dice = DiceMetric(include_background=True, reduction="mean")
    metric_dice.reset()
    metric_BCEL = BCEWithLogitsLoss()
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            BCEL += metric_BCEL(y, preds).sum()
            iou_score += metric_largest_box_iou(y.cpu(), preds.cpu())
            metric_dice(y, preds)
            metric_hd(y, preds)

    dice_score = metric_dice.aggregate().item()
    hausdorff_distance = metric_hd.aggregate().item()
    print(
        f"Epoch: {epoch}, Acc: {(num_correct/num_pixels):.2f}, and Dice score: {(dice_score):.2f}, IoU: {(iou_score/n):.2f}, hd: {(hausdorff_distance):.2f}"
    )
    return {"accuracy": tensor_to_float(num_correct/num_pixels), "bcel": tensor_to_float(BCEL), "dice": dice_score, "iou": tensor_to_float(iou_score/n), "hd": hausdorff_distance}