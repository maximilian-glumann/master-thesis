"""author: Alexander Ortlieb"""

import torch
from xray_fov.nn.boundingbox import get_largest_segments_bounding_box
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import copy

def calculate_iou_over_under_with_tolerance(model, test_image, test_mask, tolerance, device="cuda"):
    x = test_image.to(device)
    preds = model(x)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    gt_box = get_largest_segments_bounding_box(test_mask)[0]
    pred_box = get_largest_segments_bounding_box(preds.cpu())[0]

    # Coordinates for the predicted box and ground truth box
    x1_pred, y1_pred, x2_pred, y2_pred = pred_box
    x1_gt, y1_gt, x2_gt, y2_gt = gt_box

    # Add tolerance
    x1_pred, y1_pred, x2_pred, y2_pred = x1_pred - tolerance, y1_pred - tolerance, x2_pred + tolerance, y2_pred + tolerance

    # Calculate area of predicted box and ground truth box
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    
    # Calculate intersection coordinates
    x1_int = max(x1_pred, x1_gt)
    y1_int = max(y1_pred, y1_gt)
    x2_int = min(x2_pred, x2_gt)
    y2_int = min(y2_pred, y2_gt)
    
    # Check if there is an intersection
    if x1_int < x2_int and y1_int < y2_int:
        # Calculate the area of intersection
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
    else:
        # No intersection
        intersection_area = 0

    # Calculate union area
    union_area = pred_area + gt_area - intersection_area
    
    # IoU calculation
    if union_area > 0:
        iou = intersection_area / union_area
    else:
        iou = 0
    
    # Overreach calculation
    overreach = (pred_area - intersection_area) / pred_area if gt_area > 0 else 0
    
    # Underreach calculation
    underreach = (gt_area - intersection_area) / gt_area if gt_area > 0 else 0

    return overreach, underreach