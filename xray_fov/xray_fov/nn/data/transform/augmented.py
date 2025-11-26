"""authors: Alexander Ortlieb, Maximilian Glumann"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

DEFAULT_IMAGE_SIZE = 256

def transform_train(image, mask, **kwargs):
    image_size=kwargs.get("image_size",DEFAULT_IMAGE_SIZE)
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
    ret = A.Compose(
    [
        A.Resize(height=image_size, width=image_size),
        A.CropAndPad(px = 40, keep_size=True),
        ToTensorV2(),
    ],) (image=image, mask=mask)
    return ret["image"], ret["mask"]