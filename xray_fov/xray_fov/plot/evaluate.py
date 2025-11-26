"""authors: Alexander Ortlieb, Maximilian Glumann"""

import torch
from xray_fov.nn.boundingbox import get_largest_segments_bounding_box
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import copy
import numpy as np

def plot_image_mask_box_pred_box(index, model, test_image, test_mask, device="cuda", filebase=None):
    x = test_image.to(device)
    y = test_mask.to(device)
    preds = model(x)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    boxes_mask = get_largest_segments_bounding_box(test_mask)[0]
    boxes_pred = get_largest_segments_bounding_box(preds.cpu())[0]

    rect_pred = patches.Rectangle((boxes_pred[0], boxes_pred[1]), boxes_pred[2] - boxes_pred[0],
                                  boxes_pred[3] - boxes_pred[1], linewidth=4, edgecolor='orange', facecolor='none', label='predicted FOV')
    rect_mask = patches.Rectangle((boxes_mask[0], boxes_mask[1]), boxes_mask[2] - boxes_mask[0],
                                  boxes_mask[3] - boxes_mask[1], linewidth=4, edgecolor='dodgerblue', facecolor='none', label='optimal FOV')

    fig, axe = plt.subplots(figsize=(3,3))
    axe.imshow(test_image[0, 0], cmap='gray')
    pred_rect_copy = copy(rect_pred)
    mask_rect_copy = copy(rect_mask)
    axe.add_patch(mask_rect_copy)
    axe.add_patch(pred_rect_copy)
    axe.axis('off')
    axe.legend()
    fig.savefig(filebase+f"{index}_image.svg")
    plt.show()

    fig, axe = plt.subplots(figsize=(2.5,2.5))
    axe.imshow(test_mask[0, 0], cmap='gray')
    pred_rect_copy = copy(rect_pred)
    mask_rect_copy = copy(rect_mask)
    axe.add_patch(mask_rect_copy)
    axe.add_patch(pred_rect_copy)
    axe.axis('off')
    fig.savefig(filebase+f"{index}_mask.svg")
    plt.show()

    fig, axe = plt.subplots(figsize=(2.5,2.5))
    axe.imshow(preds[0, 0].cpu(), cmap='gray')
    pred_rect_copy = copy(rect_pred)
    mask_rect_copy = copy(rect_mask)
    axe.add_patch(mask_rect_copy)
    axe.add_patch(pred_rect_copy)
    axe.axis('off')
    fig.savefig(filebase+f"{index}_preds.svg")
    plt.show()
    
    #ax1.legend(loc='lower left', fontsize = 16)
    
def plot_image_mask_box_pred_box_tolerance(model, test_image, test_mask, tolerance, device="cuda"):
    x = test_image.to(device)
    y = test_mask.to(device)
    preds = model(x)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    boxes_mask = get_largest_segments_bounding_box(test_mask)[0]
    boxes_pred = get_largest_segments_bounding_box(preds.cpu())[0]
    ################################################
    # Coordinates for the predicted box and ground truth box
    x1_pred, y1_pred, x2_pred, y2_pred = boxes_pred
    x1_gt, y1_gt, x2_gt, y2_gt = boxes_mask

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
    
    # Overreach calculation
    overreach = (pred_area - intersection_area) / gt_area if gt_area > 0 else 0
    
    # Underreach calculation
    underreach = (gt_area - intersection_area) / gt_area if gt_area > 0 else 0
    ################################################

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    rect_pred = patches.Rectangle((x1_pred, y1_pred), x2_pred - x1_pred,
                                  y2_pred - y1_pred, linewidth=4, edgecolor='orange', facecolor='none', label='predicted crop with tolerance')
    rect_mask = patches.Rectangle((boxes_mask[0], boxes_mask[1]), boxes_mask[2] - boxes_mask[0],
                                  boxes_mask[3] - boxes_mask[1], linewidth=4, edgecolor='dodgerblue', facecolor='none', label='original crop')

    axs[0].imshow(test_image[0, 0], cmap='gray')
    axs[0].set_title('X-ray image', fontsize=20)

    axs[1].imshow(test_mask[0, 0], cmap='gray')
    axs[1].set_title('Original mask', fontsize=20)

    axs[2].imshow(preds[0, 0].cpu(), cmap='gray')
    axs[2].set_title('Predicted mask', fontsize=20)

    for i in [0, 1, 2]:
        pred_rect_copy = copy(rect_pred)
        mask_rect_copy = copy(rect_mask)
        axs[i].add_patch(mask_rect_copy)
        axs[i].add_patch(pred_rect_copy)
    try:
        fig.suptitle(f"Overreach: {overreach:.3f}, Underreach: {underreach:.3f}", fontsize=30)
    except:
        pass
    plt.legend()
    plt.show()
    #plt.close(fig)
    return fig
    
def plot_fov_reduction(reduced_areas, file):
    fig, axe = plt.subplots(figsize=(3.5,2.5))
    axe.hist(reduced_areas, bins = 50, range=(-10,100))
    axe.set_xlabel('Field of view reduction in %')
    axe.set_ylabel('Number of images') 
    fig.savefig(file+".svg")
    fig.savefig(file+".jpg")
    plt.show()

def plot_over_underreach(overreachs, underreachs, file):
    fig, axe = plt.subplots(figsize=(3.5,2.5))
    axe.hist(overreachs, bins = 50, range = (0, 100))
    axe.set_xlabel('Overreach in %')
    axe.set_ylabel('Number of images')
    fig.savefig(file+"over.svg")
    fig.savefig(file+"over.jpg")
    plt.show()
             
    fig, axe = plt.subplots(figsize=(3.5,2.5))
    axe.hist(underreachs, bins = 50, range = (0, 100))
    axe.set_xlabel('Underreach in %')
    axe.set_ylabel('Number of images')
    fig.savefig(file+"under.svg")
    fig.savefig(file+"under.jpg")
    plt.show()