"""author: Alexander Ortlieb"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import copy
from torchvision.ops import (
    masks_to_boxes,
    generalized_box_iou
)

def get_bounding_box(mask):
    '''
    input: mask: Tensor[N, 1, X, Y]
    output: box: Tensor[N,4]
    '''
    try:
        box = masks_to_boxes(mask.squeeze(1))
    except:
        box = torch.zeros(mask.size()[0], 4)
    return box

def get_largest_segments_bounding_box(input):
    '''
    Input: is a torch tensor with shape [B, 1, N, M]
    Output: Bounding box coordinates shape [B, 4] with coordinates (x1, y1, height, width)
    '''
    contour_segments = torch.empty(input.shape)
    for i in range(input.shape[0]):
        seg_mask = np.uint8(input*255)[i, 0]
        _, mask = cv2.threshold(seg_mask, 127, 255, cv2.THRESH_BINARY)

        # Finden aller zusammenhängenden Komponenten
        num_labels, labels_im = cv2.connectedComponents(mask)

        # Berechnen der Flächen aller Komponenten
        areas = []
        # Starten bei 1, um den Hintergrund auszuschließen
        for label in range(0, num_labels):
            area = np.sum(labels_im == label)
            areas.append(area)

        # Finden der zwei größten Komponenten
        if len(areas) < 2:
            contour_segments[i] = torch.zeros(seg_mask.shape)
        elif len(areas) < 3:
            biggest_labels = sorted(
                range(len(areas)), key=lambda sub: areas[sub])[-3:-1]
            index_mask = (labels_im == biggest_labels[0])
            contour_image = index_mask.astype(int)
            contour_image = torch.Tensor(contour_image).unsqueeze(0)
            contour_segments[i] = contour_image
        else:
            biggest_labels = sorted(
                range(len(areas)), key=lambda sub: areas[sub])[-3:-1]
            index_mask = np.logical_or(
                labels_im == biggest_labels[0], labels_im == biggest_labels[1])
            contour_image = index_mask.astype(int)
            contour_image = torch.Tensor(contour_image).unsqueeze(0)
            contour_segments[i] = contour_image
    boxes = get_bounding_box(contour_segments)
    return boxes

def metric_largest_box_iou(pred, mask):
    '''
    input: pred and mask of model: Tensor[N, 1, X, Y]
    output: sum of box_iou Tensor
    '''
    box_1 = get_largest_segments_bounding_box(pred)
    box_2 = get_largest_segments_bounding_box(mask)
    box_iou = generalized_box_iou(box_1, box_2)
    box_iou = torch.nan_to_num(box_iou, 0)
    box_iou = torch.trace(box_iou)
    return box_iou
