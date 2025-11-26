"""author: Alexander Ortlieb"""

import torch
from xray_fov.nn.boundingbox import get_largest_segments_bounding_box

def calculate_cropped_area_with_tolerance(model, test_image, test_mask, tolerance, device="cuda"):
    x = test_image.to(device)
    preds = model(x)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    gt_box = get_largest_segments_bounding_box(test_mask)[0]
    pred_box = get_largest_segments_bounding_box(preds.cpu())[0]
    
    # Coordinates for the predicted box and ground truth box
    x1_pred, y1_pred, x2_pred, y2_pred = pred_box

    # Add tolerance
    x1_pred, y1_pred, x2_pred, y2_pred = x1_pred - tolerance, y1_pred - tolerance, x2_pred + tolerance, y2_pred + tolerance

    # Calculate area of predicted box and ground truth box
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)

    # 40 px / 256 px padding per side in transform -> 176/256 = effective area
    total_area = preds.numel()*((176/256)**2)

    return pred_area / total_area