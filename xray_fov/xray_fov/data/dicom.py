"""authors: Alexander Ortlieb, Maximilian Glumann"""

import pydicom
import numpy as np
import os
import math
from xray_fov.data.operable_data import OperableData

def numpy_lstsq(xs, ys):
    A = np.vstack([xs, np.ones(len(xs))]).T
    return np.linalg.lstsq(A, ys)[0]

def noise(image, size, range): 
    cropped = image[range[1]:range[1]+size[1], range[0]:range[0]+size[0]]
    mean = round(cropped.mean(), 2)
    std = round(cropped.std(), 2)
    return cropped, mean, std

def min_noise(image, size):
    width = size[1]
    height = size[0]
    inc_x = 100
    inc_y = 100
    import math
    min_mean = math.inf
    min_std = math.inf
    min_cropped = [[]]
    for x in range(0, image.shape[0]-width, inc_x):
        for y in range(0, image.shape[1]-height, inc_y):
            cropped, mean, std = noise(image, size, (x,y))
            if(std < min_std):
                min_std = std
                min_mean = mean
                min_cropped = cropped
    return min_cropped, min_mean, min_std

def dicom_thinning(dicom, sim_muas=600, eln_var=0):
    return np.random.binomial(np.clip(dicom["photons"], 0, None).astype(int) + eln_var, sim_muas/dicom["exposure_muas"]) 
    + np.random.poisson((1-(sim_muas/dicom["exposure_muas"]))*eln_var, dicom["photons"].shape) - eln_var

class SingleDicomOperations():
    def window_min(self, op_data):
        return op_data["window_center"] - op_data["window_width"]/2
    def window_max(self, op_data):
        return op_data["window_center"] + op_data["window_width"]/2
    def q_min(self, op_data):
        return np.quantile(op_data["pixel_array"], 0.002)
    def q_max(self, op_data):
        return np.quantile(op_data["pixel_array"], 0.999)
    def muas_lin(self, op_data):
        return (op_data["clamped_pixel_array"]-op_data.parent["muas_c"])/op_data.parent["muas_m"]
    def muas_log(self, op_data):
        return np.log(np.clip(op_data["clamped_pixel_array"], 30, None) / op_data.parent["muas_a"]) / op_data.parent["muas_b"]
    def muas(self, op_data):
        muas = op_data["muas_log"]
        return muas
    def photons(self, op_data):
        temp = op_data["muas"] + np.random.normal(0, math.sqrt(abs(op_data.parent["noise_c"])), op_data["muas"].shape)
        return temp / op_data.parent["noise_m"]
    def thinned(self, op_data):
        return dicom_thinning(op_data)
    def clamped_pixel_array(self, op_data):
        return np.clip(op_data["pixel_array"], op_data["window_min"], op_data["window_max"])
    def grayscale(self, op_data):
        return (255*op_data["grayscale_float"]).astype(int)
    def grayscale_float(self, op_data):
        return (op_data["clamped_pixel_array"] - op_data["window_min"]) / op_data["window_width"]

def dicom_lstsq(dicoms):
    return numpy_lstsq(dicoms["exposure_muas"], dicoms["window_min"])[0], numpy_lstsq(dicoms["exposure_muas"], dicoms["window_max"])[1]
    
class MultipleDicomOperations():
    def muas_m(self, op_data):
        return dicom_lstsq(op_data)[0]
    def muas_c(self, op_data):
        return dicom_lstsq(op_data)[1]
    def muas_a(self, op_data):
        temp1 = op_data[1]
        x1 = temp1["exposure_muas"]
        y1 = max(temp1["window_min"], 40)
        return y1/np.exp(op_data["muas_b"]*x1)
    def muas_b(self, op_data):
        temp1 = op_data[1]
        x1 = temp1["exposure_muas"]
        y1 = max(temp1["window_min"], 40)
        temp2 = op_data[-1]
        x2 = temp2["exposure_muas"]
        y2 = max(temp2["window_min"], 40)
        return np.log(y1/y2)/(x1-x2)
    def noise_m(self, op_data):
        stds, means = op_data["noise_stats"]
        return numpy_lstsq(means, [std**2 for std in stds])[0]
    def noise_c(self, op_data):
        stds, means = op_data["noise_stats"]
        return numpy_lstsq(means, [std**2 for std in stds])[1]
    def noise_stats(self, op_data, kind="muas"):
        stds = []
        means = []
        for img in op_data[kind]:
            if op_data["noise_patch_range"] is not None:
                _, mean, std = noise(img, op_data["noise_patch_size"], op_data["noise_patch_range"])
            else:
                _, mean, std = min_noise(img, op_data["noise_patch_size"])
            stds += [std]
            means += [mean]
        return stds, means
    def __getattr__(self, key):
        if "_ipython_" in key or "_repr_" in key:
            raise AttributeError('no such data or operation!')
        data = lambda op_data : op_data.foreachByOps("SingleDicomOperations", key)
        return lambda op_data : list(data(op_data).values())

def dicom_file(file_path, caching=True):
    ds = pydicom.dcmread(file_path, force = True)
    add_dict = {
        "file_path" :     file_path.split("/")[-1],
        "exposure_muas" : ds.ExposureInuAs,
        "filter_type" :   ds.FilterType,
        "window_center" : ds.WindowCenter,
        "window_width" :  ds.WindowWidth,
        "pixel_array" :   ds.pixel_array
    }
    return OperableData(add_dict, SingleDicomOperations(), caching)    

def dicom_directory(dicom_path, noise_patch_size=(400,400), noise_patch_range=None, caching=True, add_data={}):
    image_information = []
    for index,file in enumerate(os.listdir(dicom_path)):
        file_path = os.path.join(dicom_path, file)
        addition = dicom_file(file_path)
        image_information += [addition]
    return OperableData({key:val for key,val in enumerate(sorted(image_information, key=lambda elem : elem["exposure_muas"]))} | {"len": len(image_information), "noise_patch_size": noise_patch_size, "noise_patch_range": noise_patch_range} | add_data, MultipleDicomOperations(), caching)