"""authors: Maximilian Glumann"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from skimage.exposure import match_histograms
import torch
from xray_fov.data.dicom import dicom_file

DEFAULT_IMAGE_SIZE = 256
DEFAULT_STD = 13

def transform_train(image, mask, **kwargs):
    image_size=kwargs.get("image_size",DEFAULT_IMAGE_SIZE)
    image = match_histograms(image, dicom_file(kwargs.get("reference"))["grayscale"])
    ret = A.Compose(
    [
        A.Resize(height=image_size, width=image_size),
        A.CropAndPad(px = 40, keep_size=True),
        A.Rotate(limit= 15, p = 0.5),
        ToTensorV2(),
    ]) (image=image, mask=mask)
    gaussian_noise = torch.normal(0, kwargs.get("std",DEFAULT_STD), ret["image"].shape)
    x = ret["image"]
    return (x.max()-np.random.poisson(x.max()-x+x.min())+x.min())+gaussian_noise, ret["mask"]

def transform_test(image, mask, **kwargs):
    image_size=kwargs.get("image_size",DEFAULT_IMAGE_SIZE)
    image = match_histograms(image, dicom_file(kwargs.get("reference"))["grayscale"])
    ret = A.Compose(
    [
        A.Resize(height=image_size, width=image_size),
        A.CropAndPad(px = 40, keep_size=True),
        ToTensorV2(),
    ],) (image=image, mask=mask)
    gaussian_noise = torch.normal(0, kwargs.get("std",DEFAULT_STD), ret["image"].shape)
    x = ret["image"]
    return (x.max()-np.random.poisson(x.max()-x+x.min())+x.min())+gaussian_noise, ret["mask"]