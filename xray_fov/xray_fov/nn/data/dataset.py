"""authors: Alexander Ortlieb, Maximilian Glumann"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class LungDataset(Dataset):
    def __init__(self, dir_images, dir_masks, transform, **kwargs):
        self.transform = transform
        self.dir_images = dir_images
        self.dir_masks = dir_masks
        self.images = sorted(os.listdir(dir_images))
        self.masks = sorted(os.listdir(dir_masks))
        assert(len(self.images)==len(self.masks))
        self.len = len(self.images)
        self.kwargs = kwargs

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir_images, self.images[idx])
        mask_path = os.path.join(self.dir_masks, self.masks[idx])
        image = np.array(Image.open(img_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))
        if self.transform is not None:
            image, mask = self.transform(image, mask, **self.kwargs)
        return image.to(torch.float)/255, mask.unsqueeze(0)/255
