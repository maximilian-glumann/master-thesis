"""author: Maximilian Glumann"""

from dataclasses import dataclass
import torch
from xray_fov.nn.models.utils import create_model, create_optimizer

@dataclass
class Model:
    net:torch.nn.Module
    hash:str

def model(name, model_dir, ldrs, hp):
    match name:
        case "custom":
            return Model(*create_model("xray_fov.models.unet", "UNet", model_dir, ldrs.data_dict, hp, in_channels=1, out_channels=1, features=(16, 32, 64, 128)))
        case "monai":
            return Model(*create_model("monai.networks.nets", "UNet", model_dir, ldrs.data_dict, hp, spatial_dims=2, in_channels=1, out_channels=1, channels=(16, 32, 64, 128), strides=(2, 2, 2)))
        case _:
            raise Exception("invalid model name!")

def optimizer(name, model, model_dir, hp):
    match name:
        case "adam":
            return create_optimizer("torch.optim", "Adam", model_dir, model.hash, model.net, lr=hp.learning_rate)
        case _:
            raise Exception("invalid optimizer name!")