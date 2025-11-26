"""author: Maximilian Glumann"""

import os
import numpy as np
from PIL import Image
from xray_fov.data.operable_data import OperableData
from xray_fov.data.dicom import dicom_thinning, noise

class SingleImageOperations():
    def pixel_array(self, op_data):
        file_path = os.path.join(op_data["file_path"], op_data["file_name"])
        return np.array(Image.open(file_path).convert('L'))
    def clamped_pixel_array(self, op_data):
        return op_data["pixel_array"]
    def grayscale(self, op_data):
        return op_data["pixel_array"]
    def exposure_muas(self, op_data):
        return 4000
    def muas(self, op_data):
        muas = (255-op_data["pixel_array"].astype(int))*op_data["exposure_muas"]/255
        return muas
    def photons(self, op_data):
        return op_data["muas"]
    def thinned(self, op_data, muas=10, eln_var=0):
        return dicom_thinning(op_data, muas, eln_var)

class MultipleImageOperations():
    def __getattr__(self, key):
        if "_ipython_" in key or "_repr_" in key:
            raise AttributeError('no such data or operation!')
        data = lambda op_data : op_data.foreachByOps("SingleImageOperations", key)
        return lambda op_data : list(data(op_data).values())

# variant without OperableData because of PyTorch issue:
# https://github.com/pytorch/pytorch/issues/13246
def thinned_image(image, sim_muas, eln_var):
    photons = (255-image.astype(int))*4000/255
    return np.random.binomial(np.clip(photons, 0, None).astype(int) + eln_var, sim_muas/4000) 
    + np.random.poisson((1-(sim_muas/4000))*eln_var, image.shape) - eln_var

    
def image_directory(image_path, preload=True, caching=True):
    image_information = []
    for index,file in enumerate(os.listdir(image_path)):
        add_dict = {
            "file_path" :     image_path,
            "file_name" :     file,
        }
        addition = OperableData(add_dict, SingleImageOperations(), caching)
        if preload:
            addition.pixel_array()
        image_information += [addition]
    return OperableData({key:val for key,val in enumerate(image_information)} | {"len": len(image_information)}, MultipleImageOperations(), caching)