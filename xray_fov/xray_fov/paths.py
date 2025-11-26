"""author: Maximilian Glumann"""

import os
data_path = os.getenv("HOME")+'/master-thesis/data'
data_path_alex = data_path+'/alex'

paths = {
    "dicom": {
        "phantom-alex": data_path_alex+'/low-dose-simulation/phantom',
        "phantom": data_path+'/2025-09-09-xray/phantom',
        "phantom-noxray": data_path+'/2025-09-09-xray/phantom-noxray',
        "air-noxray": data_path+'/2025-09-09-xray/air-noxray',
        "air": data_path+'/2025-09-09-xray/air',
    },
    "nn": {
        "train-images": data_path_alex+'/segmentation/data/train/images',
        "train-masks": data_path_alex+'/segmentation/data/train/masks',
        "test-images": data_path_alex+'/segmentation/data/test/images',
        "test-masks": data_path_alex+'/segmentation/data/test/masks',
    }
}

output_path = data_path+'/output'
outs = {
    "phantom-air-noxray": output_path+'/phantom-air-noxray',
    "single-phantoms": output_path+'/single-phantoms',
    "nn-images" : output_path+'/nn-images',
    "models": output_path+'/models',
    "phantom-stats": output_path+'/phantom-stats',
    "phantom-noise-stats": output_path+'/phantom-noise-stats',
    "dicom-convert": output_path+'/dicom-convert',
    "dicom-thinning": output_path+'/dicom-thinning',
    "nn-thinning": output_path+'/nn-thinning',
    "nn-thinning-eval": output_path+'/nn-thinning-eval',
    "histogram-matching": output_path+'/histogram-matching',
    "nn-eval": output_path+'/nn-eval',
}

from pathlib import Path
for path in outs.values():
    Path(path).mkdir(parents=True, exist_ok=True)