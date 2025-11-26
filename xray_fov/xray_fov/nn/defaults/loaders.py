"""author: Maximilian Glumann"""

from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Dict
from xray_fov.nn.models.utils import create_loaders

@dataclass
class Loaders:
    train:DataLoader
    test:DataLoader
    validation:DataLoader
    data_dict:Dict

def loaders(name, data_dirs, reference_path, hp):
    match name:
        case "augmented":
             return Loaders(*create_loaders(
                data_dirs, "augmented",
                num_workers=hp.num_workers, pin_memory=hp.pin_memory, batch_size=hp.batch_size ))
        case "gaussian_ratio":
             return Loaders(*create_loaders(
                data_dirs, "gaussian_ratio", image_std=10, noise_std_ratio=0.5,
                num_workers=hp.num_workers, pin_memory=hp.pin_memory, batch_size=hp.batch_size ))
        case "gaussian":
             return Loaders(*create_loaders(
                data_dirs, "gaussian", reference=reference_path, std=10,
                num_workers=hp.num_workers, pin_memory=hp.pin_memory, batch_size=hp.batch_size ))
        case "poisson_gaussian":
             return Loaders(*create_loaders(
                data_dirs, "poisson_gaussian", reference=reference_path, std=10,
                num_workers=hp.num_workers, pin_memory=hp.pin_memory, batch_size=hp.batch_size ))
        case "poisson_inv_gaussian":
             return Loaders(*create_loaders(
                data_dirs, "poisson_inv_gaussian", reference=reference_path, std=10,
                num_workers=hp.num_workers, pin_memory=hp.pin_memory, batch_size=hp.batch_size ))
        case "poisson_inv":
             return Loaders(*create_loaders(
                data_dirs, "poisson_inv", reference=reference_path,
                num_workers=hp.num_workers, pin_memory=hp.pin_memory, batch_size=hp.batch_size ))
        case "poisson":
             return Loaders(*create_loaders(
                data_dirs, "poisson", reference=reference_path,
                num_workers=hp.num_workers, pin_memory=hp.pin_memory, batch_size=hp.batch_size ))
        case "thinned":
            return Loaders(*create_loaders(
                data_dirs, "thinned", sim_muas=50, eln_var=0,
                num_workers=hp.num_workers, pin_memory=hp.pin_memory, batch_size=hp.batch_size ))       
        case _:
            raise Exception("invalid loader name!")