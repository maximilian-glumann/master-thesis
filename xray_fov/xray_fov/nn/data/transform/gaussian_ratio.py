"""authors: Alexander Ortlieb, Maximilian Glumann"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from skimage.exposure import match_histograms
import torch

DEFAULT_IMAGE_SIZE = 256
DEFAULT_IMAGE_STD = 13
DEFAULT_NOISE_STD_RATIO = 2

def transform_train(image, mask, **kwargs):
    image_size=kwargs.get("image_size",DEFAULT_IMAGE_SIZE)
    s = np.random.normal(130, kwargs.get("image_std",DEFAULT_IMAGE_STD), (2995, 2993)).astype(int)
    image = match_histograms(image, s)
    ret = A.Compose(
    [
        A.Resize(height=image_size, width=image_size),
        A.CropAndPad(px = 40, keep_size=True),
        A.Rotate(limit= 15, p = 0.5),
        ToTensorV2(),
    ]) (image=image, mask=mask)
    gaussian_noise = torch.normal(0, kwargs.get("image_std",DEFAULT_IMAGE_STD) / kwargs.get("noise_std_ratio",DEFAULT_NOISE_STD_RATIO), ret["image"].shape)
    return ret["image"]+gaussian_noise, ret["mask"]

def transform_test(image, mask, **kwargs):
    image_size=kwargs.get("image_size",DEFAULT_IMAGE_SIZE)
    s = np.random.normal(130, kwargs.get("image_std",DEFAULT_IMAGE_STD), (2995, 2993)).astype(int)
    image = match_histograms(image, s)
    ret = A.Compose(
    [
        A.Resize(height=image_size, width=image_size),
        A.CropAndPad(px = 40, keep_size=True),
        ToTensorV2(),
    ],) (image=image, mask=mask)
    gaussian_noise = torch.normal(0, kwargs.get("image_std",DEFAULT_IMAGE_STD) / kwargs.get("noise_std_ratio",DEFAULT_NOISE_STD_RATIO), ret["image"].shape)
    return ret["image"]+gaussian_noise, ret["mask"]