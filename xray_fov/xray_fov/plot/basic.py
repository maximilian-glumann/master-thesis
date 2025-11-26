"""author: Maximilian Glumann"""

import numpy as np

def plot(ax, xs, ys, label, color=None, log=False):
    if log:
        ax.loglog(xs,ys,label=label, color=color)
    else:
        ax.plot(xs,ys,label=label, color=color)

def plot_lstsq(ax, xs, ys, label, color, log=False):
    A = np.vstack([xs, np.ones(len(xs))]).T
    m, c = np.linalg.lstsq(A, ys)[0]
    if log:
        ax.loglog(0, c, 'x', color=color)
        ax.loglog(xs, [m*x+c for x in xs], label=label, color=color)
    else:
        ax.plot(0, c, 'x', color=color)
        ax.plot(xs, [m*x+c for x in xs], label=label, color=color)
    
def plot_both(ax, xs, ys, label, color, log=False):
    plot(ax,xs,ys,None,color,log)
    plot_lstsq(ax,xs,ys,label,color,log)