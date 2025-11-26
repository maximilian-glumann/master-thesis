"""authors: Alexander Ortlieb, Maximilian Glumann"""

import numpy as np
import matplotlib.pyplot as plt
from xray_fov.plot.basic import plot, plot_lstsq, plot_both
from xray_fov.data.dicom import noise, min_noise

def plot_noise_stats(phantoms, kind, plot_exposures=False, plot_log=False, plot_lstsq=True):
    exposures = phantoms["exposure_muas"]
    stds, means = phantoms.noise_stats(kind)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    if plot_exposures:
        xs = exposures
        ax.set_xlabel("exposure [$\\mu As$]")
    else:
        xs = means
        ax.set_xlabel("noise patch mean")

    ax.set_ylabel("noise patch variance")
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,2))
    ax.grid(True)
    
    if plot_lstsq:
        plot_both(ax, xs, [std**2 for std in stds], "var", None, plot_log)
    else:
        plot(ax, xs, [std**2 for std in stds], "var", None, plot_log)

    plt.close()
    return fig

def plot_noise_area(phantoms, kind, show_noise=True, filebase=None):
    if show_noise:
        size = phantoms["noise_patch_size"]
        range = phantoms["noise_patch_range"]
    vmin = np.min([np.min(x) for x in phantoms[kind]])
    vmax = np.max([np.max(x) for x in phantoms[kind]])
    
    for phantom, exposure in zip(phantoms[kind], phantoms["exposure_muas"]):
        if show_noise:
            if range is not None:
                cropped, mean, std = noise(phantom, size, range)
            else:
                cropped, mean, std = min_noise(phantom, size)
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
        ax1.axis('off')
        ax2.axis('off')
        im = ax1.imshow(phantom, cmap='gray', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax1)
        ax1.set_title(f'mean: {round( phantom.mean(),2)}, std: {round(phantom.std(),2)}')
        if show_noise:
            im2 = ax2.imshow(cropped, cmap='gray', vmin=vmin, vmax=vmax)
            fig.colorbar(im2, ax=ax2)
            ax2.set_title(f'mean: {round(mean,2)}, std: {round(std,2)}')
        if filebase is not None:
            fig.savefig(filebase+f"_{exposure}.svg")
            fig.savefig(filebase+f"_{exposure}.jpg")
        plt.show()
    return