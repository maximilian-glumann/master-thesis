"""author: Maximilian Glumann"""

from IPython.display import display_markdown
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from xray_fov.data.dicom import noise

def plot_dicom(dicom, filebase=None):
    display_markdown(f"## {dicom["exposure_muas"]} | clamped_pixel_array", raw=True)
    fig, axe = plt.subplots(figsize=(7, 6))
    im = axe.imshow(dicom["clamped_pixel_array"], cmap="gray")
    axe.axis('off')
    fig.colorbar(im, ax=axe)

    if filebase is not None:
        fig.savefig(filebase+f"_{dicom["exposure_muas"]}_clamped.svg")
        fig.savefig(filebase+f"_{dicom["exposure_muas"]}_clamped.jpg")
    plt.show()

    display_markdown(f"## {dicom["exposure_muas"]} | muas", raw=True)
    fig, axe = plt.subplots(figsize=(4, 3))
    im = axe.imshow(dicom["muas"], cmap="gray")
    fig.colorbar(im, ax=axe)
    axe.axis('off')

    if filebase is not None:
        fig.savefig(filebase+f"_{dicom["exposure_muas"]}_muas.svg")
        fig.savefig(filebase+f"_{dicom["exposure_muas"]}_muas.jpg")
    plt.show()

    display_markdown(f"## {dicom["exposure_muas"]} | photons", raw=True)
    fig, axe = plt.subplots(figsize=(4, 3))
    im = axe.imshow(dicom["photons"], cmap="gray")
    fig.colorbar(im, ax=axe)
    axe.axis('off')

    if filebase is not None:
        fig.savefig(filebase+f"_{dicom["exposure_muas"]}_photons.svg")
        fig.savefig(filebase+f"_{dicom["exposure_muas"]}_photons.jpg")
    plt.show()

def plot_dicom_multiple(dicoms, filebase=None):
    for _, dicom in dicoms.getByOps("SingleDicomOperations"):
        plot_dicom(dicom, filebase)    