"""author: Maximilian Glumann"""

from xray_fov.plot.basic import plot, plot_lstsq, plot_both
import matplotlib.pyplot as plt
import numpy as np

def plot_phantom_stats(phantoms, noxray=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = phantoms["exposure_muas"]

    plot(ax, xs, phantoms["window_min"], "window_min")
    plot(ax, xs, phantoms["window_max"], "window_max")

    if not noxray:
        a = phantoms["muas_a"]
        b = phantoms["muas_b"]
        xxs = np.linspace(0, 4000, 100)
        ys = a * np.exp(b * xxs)
        ax.plot(xxs, ys, label="$a\\cdot\\exp(b\\cdot x)$")
        
    ax.set_xlabel("exposure [$\\mu As$]")
    ax.set_ylabel("pixel array value")
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,3))
    ax.grid(True)

    ax.legend()
    
    plt.close(fig)
    return fig