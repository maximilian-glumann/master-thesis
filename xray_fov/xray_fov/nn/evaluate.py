"""authors: Alexander Ortlieb, Maximilian Glumann"""

from IPython.display import display_markdown
from xray_fov.plot.evaluate import plot_image_mask_box_pred_box, plot_fov_reduction, plot_over_underreach
from xray_fov.nn.over_under_reach import calculate_iou_over_under_with_tolerance
from xray_fov.nn.fov import calculate_cropped_area_with_tolerance
from xray_fov.nn.models.utils import create_model_from_hash, create_loaders_from_hash, load_model
from xray_fov.nn.defaults.hyperparameters import hyperparameters as hp
from xray_fov.paths import outs
import numpy as np

def evaluate(hash, filebase=None):
    net = create_model_from_hash(hash, outs["models"]).to(hp.training.device)
    load_model(net, hash, outs["models"])
    train_loader, test_loader, validation_loader = create_loaders_from_hash(hash, outs["models"], batch_size=1)

    for (index, (image, target)) in enumerate(test_loader):
        display_markdown(f"# index: {index}", raw=True)
        plot_image_mask_box_pred_box(index, net, image, target, device=hp.training.device, filebase=filebase)

def evaluate_fov(hash, tolerance, filebase=None):
    net = create_model_from_hash(hash, outs["models"]).to(hp.training.device)
    load_model(net, hash, outs["models"])
    train_loader, test_loader, validation_loader = create_loaders_from_hash(hash, outs["models"], batch_size=1)
    
    reduced_areas = 100*np.array([1 - calculate_cropped_area_with_tolerance(net, image, mask, tolerance = tolerance, device=hp.training.device) for image, mask in test_loader])
    plot_fov_reduction(reduced_areas, filebase+f"{tolerance}")
    return reduced_areas[0], np.mean(reduced_areas), np.std(reduced_areas), reduced_areas

def evaluate_over_underreach(hash, tolerance, filebase=None):
    net = create_model_from_hash(hash, outs["models"]).to(hp.training.device)
    load_model(net, hash, outs["models"])
    train_loader, test_loader, validation_loader = create_loaders_from_hash(hash, outs["models"], batch_size=1)
    
    overreachs, underreachs = [], []
    sum_maks, sum_image = 0, 0
    for image, mask in test_loader:
        overreach, underreach = calculate_iou_over_under_with_tolerance(net, image, mask, tolerance=tolerance, device=hp.training.device)
        overreachs.append(overreach)
        underreachs.append(underreach)
        sum_maks += (mask == 1).sum()
        sum_image += 256*256
        
    overreachs = 100*np.array(overreachs)
    underreachs = 100*np.array(underreachs)

    plot_over_underreach(overreachs, underreachs, filebase+f"{tolerance}_")