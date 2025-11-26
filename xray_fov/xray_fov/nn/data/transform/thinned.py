"""author: Maximilian Glumann"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from xray_fov.data.image import thinned_image
# thinned_image without OperableData because of PyTorch issue:
# https://github.com/pytorch/pytorch/issues/13246

DEFAULT_IMAGE_SIZE = 256
DEFAULT_SIM_MUAS = 50
DEFAULT_ELN_VAR = 0

def transform_train(image, mask, **kwargs):
    image_size=kwargs.get("image_size",DEFAULT_IMAGE_SIZE)
    image = thinned_image(image, sim_muas=kwargs.get("sim_muas",DEFAULT_SIM_MUAS), eln_var=kwargs.get("eln_var",DEFAULT_ELN_VAR)).astype(float)
    ret = A.Compose(
    [
        A.Resize(height=image_size, width=image_size),
        A.CropAndPad(px = 40, keep_size=True),
        A.Rotate(limit= 15, p = 0.5),
        ToTensorV2(),
    ]) (image=image, mask=mask)
    return ret["image"], ret["mask"]

def transform_test(image, mask, **kwargs):
    image_size=kwargs.get("image_size",DEFAULT_IMAGE_SIZE)
    image = thinned_image(image, sim_muas=kwargs.get("sim_muas",DEFAULT_SIM_MUAS), eln_var=kwargs.get("eln_var",DEFAULT_ELN_VAR)).astype(float)
    ret = A.Compose(
    [
        A.Resize(height=image_size, width=image_size),
        A.CropAndPad(px = 40, keep_size=True),
        ToTensorV2(),
    ],) (image=image, mask=mask)
    return ret["image"], ret["mask"]